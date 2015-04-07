// Copyright 2014 Arjan Topolovec
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "bencode"]
#![crate_type = "rlib"]
#![crate_type = "dylib"]

/*!
  Bencode parsing and serialization

  # Encoding

  ## Using `Encodable`

  ```rust
  extern crate rustc_serialize;
  extern crate bencode;

  use rustc_serialize::Encodable;

  use bencode::encode;

  #[derive(RustcEncodable)]
  struct MyStruct {
      string: String,
      id: usize,
  }

  fn main() {
      let s = MyStruct { string: "Hello bencode".to_string(), id: 1 };
      let result: Vec<u8> = encode(&s).unwrap();
  }
  ```

  ## Using `ToBencode`

  ```rust
  extern crate bencode;

  use std::collections::BTreeMap;

  use bencode::{Bencode, ToBencode};
  use bencode::util::ByteString;

  struct MyStruct {
      a: isize,
      b: String,
      c: Vec<u8>,
  }

  impl ToBencode for MyStruct {
      fn to_bencode(&self) -> bencode::Bencode {
          let mut m = BTreeMap::new();
          m.insert(ByteString::from_str("a"), self.a.to_bencode());
          m.insert(ByteString::from_str("b"), self.b.to_bencode());
          m.insert(ByteString::from_str("c"), Bencode::ByteString(self.c.to_vec()));
          Bencode::Dict(m)
      }
  }

  fn main() {
      let s = MyStruct{ a: 5, b: "foo".to_string(), c: vec![1, 2, 3, 4] };
      let bencode: bencode::Bencode = s.to_bencode();
      let result: Vec<u8> = bencode.to_bytes().unwrap();
  }

  ```

  # Decoding

  ## Using `Decodable`

  ```rust
  extern crate rustc_serialize;
  extern crate bencode;

  use rustc_serialize::{Encodable, Decodable};

  use bencode::{encode, Decoder};

  #[derive(RustcEncodable, RustcDecodable, PartialEq)]
  struct MyStruct {
      a: i32,
      b: String,
      c: Vec<u8>,
  }

  fn main() {
      let s = MyStruct{ a: 5, b: "foo".to_string(), c: vec![1, 2, 3, 4] };
      let enc: Vec<u8> = encode(&s).unwrap();

      let bencode: bencode::Bencode = bencode::from_vec(enc).unwrap();
      let mut decoder = Decoder::new(&bencode);
      let result: MyStruct = Decodable::decode(&mut decoder).unwrap();
      assert!(s == result)
  }
  ```

  ## Using `FromBencode`

  ```rust
  extern crate bencode;

  use std::collections::BTreeMap;

  use bencode::{FromBencode, ToBencode, Bencode};
  use bencode::util::ByteString;

  #[derive(PartialEq)]
  struct MyStruct {
      a: i32
  }

  impl ToBencode for MyStruct {
      fn to_bencode(&self) -> bencode::Bencode {
          let mut m = BTreeMap::new();
          m.insert(ByteString::from_str("a"), self.a.to_bencode());
          Bencode::Dict(m)
      }
  }

  impl FromBencode for MyStruct {
      fn from_bencode(bencode: &bencode::Bencode) -> Option<MyStruct> {
          match bencode {
              &Bencode::Dict(ref m) => {
                  match m.get(&ByteString::from_str("a")) {
                      Some(a) => FromBencode::from_bencode(a).map(|a| {
                          MyStruct{ a: a }
                      }),
                      _ => None
                  }
              }
              _ => None
          }
      }
  }

  fn main() {
      let s = MyStruct{ a: 5 };
      let enc: Vec<u8>  = s.to_bencode().to_bytes().unwrap();

      let bencode: bencode::Bencode = bencode::from_vec(enc).unwrap();
      let result: MyStruct = FromBencode::from_bencode(&bencode).unwrap();
      assert!(s == result)
  }
  ```

  ## Using Streaming Parser

  ```rust
  extern crate rustc_serialize;
  extern crate bencode;

  use bencode::streaming::BencodeEvent;
  use bencode::streaming::StreamingParser;
  use rustc_serialize::Encodable;

  use bencode::encode;

  #[derive(RustcEncodable, RustcDecodable, PartialEq)]
  struct MyStruct {
      a: i32,
      b: String,
      c: Vec<u8>,
  }

  fn main() {
      let s = MyStruct{ a: 5, b: "foo".to_string(), c: vec![2, 2, 3, 4] };
      let enc: Vec<u8> = encode(&s).unwrap();

      let mut streaming = StreamingParser::new(enc.into_iter());
      for event in streaming {
          match event {
              BencodeEvent::DictStart => println!("dict start"),
              BencodeEvent::DictEnd => println!("dict end"),
              BencodeEvent::NumberValue(n) => println!("number = {}", n),
              // ...
              _ => println!("Unhandled event: {:?}", event)
          }
      }
  }
  ```
*/

#![feature(core, old_io, std_misc, io, test)]

extern crate rustc_serialize;

use std::old_io::{self, IoResult, IoError};
use std::fmt;
use std::str;
use std::vec::Vec;
use std::num::FromStrRadix;

use rustc_serialize as serialize;
use rustc_serialize::Encodable;

use std::collections::BTreeMap;
use std::collections::HashMap;

use streaming::{StreamingParser, Error};
use streaming::BencodeEvent;
use streaming::BencodeEvent::{NumberValue, ByteStringValue, ListStart, ListEnd,
                              DictStart, DictKey, DictEnd, ParseError};
use self::Bencode::{Empty, Number, ByteString, List, Dict};
use self::DecoderError::{Message, Unimplemented, Expecting, StringEncoding};

pub mod streaming;
pub mod util;

#[inline]
fn fmt_bytestring(s: &[u8], fmt: &mut fmt::Formatter) -> fmt::Result {
  match str::from_utf8(s) {
    Ok(s) => write!(fmt, "s\"{}\"", s),
    Err(..) => write!(fmt, "s{:?}", s),
  }
}

#[derive(PartialEq, Clone, Debug)]
pub enum Bencode {
    Empty,
    Number(i64),
    ByteString(Vec<u8>),
    List(ListVec),
    Dict(DictMap),
}

impl fmt::Display for Bencode {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        format(fmt, self)
    }
}

fn format(fmt: &mut fmt::Formatter, v: &Bencode) -> fmt::Result {
    match *v {
        Bencode::Empty => { Ok(()) }
        Bencode::Number(v) => write!(fmt, "{}", v),
        Bencode::ByteString(ref v) => fmt_bytestring(v, fmt),
        Bencode::List(ref v) => {
            try!(write!(fmt, "["));
            let mut first = true;
            for value in v.iter() {
                if first {
                    first = false;
                } else {
                    try!(write!(fmt, ", "));
                }
                try!(write!(fmt, "{}", *value));
            }
            write!(fmt, "]")
        }
        Bencode::Dict(ref v) => {
            try!(write!(fmt, "{{"));
            let mut first = true;
            for (key, value) in v.iter() {
                if first {
                    first = false;
                } else {
                    try!(write!(fmt, ", "));
                }
                try!(write!(fmt, "{}: {}", *key, *value));
            }
            write!(fmt, "}}")
        }
    }
}

pub type ListVec = Vec<Bencode>;
pub type DictMap = BTreeMap<util::ByteString, Bencode>;

impl Bencode {
    pub fn to_writer(&self, writer: &mut old_io::Writer) -> old_io::IoResult<()> {
        let mut encoder = Encoder::new(writer);
        self.encode(&mut encoder)
    }

    pub fn to_bytes(&self) -> old_io::IoResult<Vec<u8>> {
        let mut writer = old_io::MemWriter::new();
        match self.to_writer(&mut writer) {
            Ok(_) => Ok(writer.into_inner()),
            Err(err) => Err(err)
        }
    }
}

impl Encodable for Bencode {
    fn encode<S: serialize::Encoder>(&self, e: &mut S) -> Result<(), S::Error> {
        match self {
            &Bencode::Empty => Ok(()),
            &Bencode::Number(v) => e.emit_i64(v),
            &Bencode::ByteString(ref v) => e.emit_str(unsafe { str::from_utf8_unchecked(v) }),
            &Bencode::List(ref v) => v.encode(e),
            &Bencode::Dict(ref v) => v.encode(e)
        }
    }
}

pub trait ToBencode {
    fn to_bencode(&self) -> Bencode;
}

pub trait FromBencode {
    fn from_bencode(&Bencode) -> Option<Self>;
}

impl ToBencode for () {
    fn to_bencode(&self) -> Bencode {
        Bencode::ByteString(Vec::new())
    }
}

impl FromBencode for () {
    fn from_bencode(bencode: &Bencode) -> Option<()> {
        match bencode {
            &Bencode::ByteString(ref v) => {
                if v.len() == 0 {
                    Some(())
                } else {
                    None
                }
            }
            _ => None
        }
    }
}

impl<T: ToBencode> ToBencode for Option<T> {
    fn to_bencode(&self) -> Bencode {
        match self {
            &Some(ref v) => v.to_bencode(),
            &None => Bencode::ByteString(b"nil".to_vec())
        }
    }
}

impl<T: FromBencode> FromBencode for Option<T> {
    fn from_bencode(bencode: &Bencode) -> Option<Option<T>> {
        match bencode {
            &Bencode::ByteString(ref v) => {
                if v == b"nil" {
                    return Some(None)
                }
            }
            _ => ()
        }
        FromBencode::from_bencode(bencode).map(|v| Some(v))
    }
}
macro_rules! derive_num_to_bencode(($t:ty) => (
    impl ToBencode for $t {
        fn to_bencode(&self) -> Bencode { Bencode::Number(*self as i64) }
    }
));

macro_rules! derive_num_from_bencode(($t:ty) => (
    impl FromBencode for $t {
        fn from_bencode(bencode: &Bencode) -> Option<$t> {
            match bencode {
                &Bencode::Number(v) => Some(v as $t),
                _ => None
            }
        }
    }
));

derive_num_to_bencode!(isize);
derive_num_from_bencode!(isize);

derive_num_to_bencode!(i8);
derive_num_from_bencode!(i8);

derive_num_to_bencode!(i16);
derive_num_from_bencode!(i16);

derive_num_to_bencode!(i32);
derive_num_from_bencode!(i32);

derive_num_to_bencode!(i64);
derive_num_from_bencode!(i64);

derive_num_to_bencode!(usize);
derive_num_from_bencode!(usize);

derive_num_to_bencode!(u8);
derive_num_from_bencode!(u8);

derive_num_to_bencode!(u16);
derive_num_from_bencode!(u16);

derive_num_to_bencode!(u32);
derive_num_from_bencode!(u32);

derive_num_to_bencode!(u64);
derive_num_from_bencode!(u64);

impl ToBencode for f32 {
    fn to_bencode(&self) -> Bencode {
        Bencode::ByteString(std::f32::to_str_hex(*self).as_bytes().to_vec())
    }
}

impl FromBencode for f32 {
    fn from_bencode(bencode: &Bencode) -> Option<f32> {
        match bencode {
            &Bencode::ByteString(ref v)  => {
                match str::from_utf8(v) {
                    Ok(s) => FromStrRadix::from_str_radix(s, 16).ok(),
                    Err(..) => None
                }
            }
            _ => None
        }
    }
}

impl ToBencode for f64 {
    fn to_bencode(&self) -> Bencode {
        Bencode::ByteString(std::f64::to_str_hex(*self).as_bytes().to_vec())
    }
}

impl FromBencode for f64 {
    fn from_bencode(bencode: &Bencode) -> Option<f64> {
        match bencode {
            &Bencode::ByteString(ref v)  => {
                match str::from_utf8(v) {
                    Ok(s) => FromStrRadix::from_str_radix(s, 16).ok(),
                    Err(..) => None
                }
            }
            _ => None
        }
    }
}

impl ToBencode for bool {
    fn to_bencode(&self) -> Bencode {
        if *self {
            Bencode::ByteString(b"true".to_vec())
        } else {
            Bencode::ByteString(b"false".to_vec())
        }
    }
}

impl FromBencode for bool {
    fn from_bencode(bencode: &Bencode) -> Option<bool> {
        match bencode {
            &Bencode::ByteString(ref v) => {
                if v == b"true" {
                    Some(true)
                } else if v == b"false" {
                    Some(false)
                } else {
                    None
                }
            }
            _ => None
        }
    }
}

impl ToBencode for char {
    fn to_bencode(&self) -> Bencode {
        Bencode::ByteString(self.to_string().as_bytes().to_vec())
    }
}

impl FromBencode for char {
    fn from_bencode(bencode: &Bencode) -> Option<char> {
        let s: Option<String> = FromBencode::from_bencode(bencode);
        s.and_then(|s| {
            if s.chars().count() == 1 {
                Some(s.chars().next().unwrap())
            } else {
                None
            }
        })
    }
}

impl ToBencode for String {
    fn to_bencode(&self) -> Bencode { Bencode::ByteString(self.as_bytes().to_vec()) }
}

impl FromBencode for String {
    fn from_bencode(bencode: &Bencode) -> Option<String> {
        match bencode {
            &Bencode::ByteString(ref v) => std::str::from_utf8(v).map(|s| s.to_string()).ok(),
            _ => None
        }
    }
}

impl<T: ToBencode> ToBencode for Vec<T> {
    fn to_bencode(&self) -> Bencode { Bencode::List(self.iter().map(|e| e.to_bencode()).collect()) }
}

impl<T: FromBencode> FromBencode for Vec<T> {
    fn from_bencode(bencode: &Bencode) -> Option<Vec<T>> {
        match bencode {
            &Bencode::List(ref es) => {
                let mut list = Vec::new();
                for e in es.iter() {
                    match FromBencode::from_bencode(e) {
                        Some(v) => list.push(v),
                        None => return None
                    }
                }
                Some(list)
            }
            _ => None
        }
    }
}

macro_rules! map_to_bencode {
    ($m:expr) => {{
        let mut m = BTreeMap::new();
        for (key, value) in $m.iter() {
            m.insert(util::ByteString::from_vec(key.as_bytes().to_vec()), value.to_bencode());
        }
        Bencode::Dict(m)
    }}
}

macro_rules! map_from_bencode {
    ($mty:ident, $bencode:expr) => {{
        let res = match $bencode {
            &Bencode::Dict(ref map) => {
                let mut m = $mty::new();
                for (key, value) in map.iter() {
                    match str::from_utf8(key.as_slice()) {
                        Ok(k) => {
                            let val: Option<T> = FromBencode::from_bencode(value);
                            match val {
                                Some(v) => m.insert(k.to_string(), v),
                                None => return None
                            }
                        }
                        Err(..) => return None
                    };
                }
                Some(m)
            }
            _ => None
        };
        res
    }}
}

impl<T: ToBencode> ToBencode for BTreeMap<String, T> {
    fn to_bencode(&self) -> Bencode {
        map_to_bencode!(self)
    }
}

impl<T: FromBencode> FromBencode for BTreeMap<String, T> {
    fn from_bencode(bencode: &Bencode) -> Option<BTreeMap<String, T>> {
        map_from_bencode!(BTreeMap, bencode)
    }
}

impl<T: ToBencode> ToBencode for HashMap<String, T> {
    fn to_bencode(&self) -> Bencode {
        map_to_bencode!(self)
    }
}

impl<T: FromBencode> FromBencode for HashMap<String, T> {
    fn from_bencode(bencode: &Bencode) -> Option<HashMap<String, T>> {
        map_from_bencode!(HashMap, bencode)
    }
}

pub fn from_buffer(buf: &[u8]) -> Result<Bencode, Error> {
    from_iter(buf.iter().map(|b| *b))
}

pub fn from_vec(buf: Vec<u8>) -> Result<Bencode, Error> {
    from_buffer(&buf[..])
}

pub fn from_iter<T: Iterator<Item=u8>>(iter: T) -> Result<Bencode, Error> {
    let streaming_parser = StreamingParser::new(iter);
    let mut parser = Parser::new(streaming_parser);
    parser.parse()
}

pub fn encode<T: serialize::Encodable>(t: T) -> IoResult<Vec<u8>> {
    let mut w = old_io::MemWriter::new();
    {
        let mut encoder = Encoder::new(&mut w);
        match t.encode(&mut encoder) {
            Err(e) => return Err(e),
            _ => {}
        }
    }
    Ok(w.into_inner())
}

macro_rules! tryenc(($e:expr) => (
    match $e {
        Ok(e) => e,
        Err(e) => {
            return
        }
    }
));

pub type EncoderResult<T> = IoResult<T>;

pub struct Encoder<'a> {
    writer: &'a mut (old_io::Writer + 'a),
    writers: Vec<old_io::MemWriter>,
    expect_key: bool,
    keys: Vec<util::ByteString>,
    is_none: bool,
    stack: Vec<BTreeMap<util::ByteString, Vec<u8>>>,
}

impl<'a> Encoder<'a> {
    pub fn new(writer: &'a mut old_io::Writer) -> Encoder<'a> {
        Encoder {
            writer: writer,
            writers: Vec::new(),
            expect_key: false,
            keys: Vec::new(),
            is_none: false,
            stack: Vec::new()
        }
    }

    fn get_writer(&mut self) -> &mut old_io::Writer {
        if self.writers.len() == 0 {
            &mut self.writer as &mut old_io::Writer
        } else {
            self.writers.last_mut().unwrap() as &mut old_io::Writer
        }
    }

    fn encode_dict(&mut self, dict: &BTreeMap<util::ByteString, Vec<u8>>) -> EncoderResult<()> {
        try!(write!(self.get_writer(), "d"));
        for (key, value) in dict.iter() {
            try!(key.encode(self));
            try!(self.get_writer().write_all(value));
        }
        write!(self.get_writer(), "e")
    }

    fn error(&mut self, msg: &'static str) -> EncoderResult<()> {
        Err(IoError {
            kind: old_io::InvalidInput,
            desc: msg,
            detail: None
        })
    }
}

macro_rules! expect_value(($slf:expr) => {
    if $slf.expect_key {
        return $slf.error("Only 'string' map keys allowed");
    }
});

impl<'a> serialize::Encoder for Encoder<'a> {
    type Error = IoError;

    fn emit_nil(&mut self) -> EncoderResult<()> { expect_value!(self); write!(self.get_writer(), "0:") }

    fn emit_usize(&mut self, v: usize) -> EncoderResult<()> { self.emit_i64(v as i64) }

    fn emit_u8(&mut self, v: u8) -> EncoderResult<()> { self.emit_i64(v as i64) }

    fn emit_u16(&mut self, v: u16) -> EncoderResult<()> { self.emit_i64(v as i64) }

    fn emit_u32(&mut self, v: u32) -> EncoderResult<()> { self.emit_i64(v as i64) }

    fn emit_u64(&mut self, v: u64) -> EncoderResult<()> { self.emit_i64(v as i64) }

    fn emit_isize(&mut self, v: isize) -> EncoderResult<()> { self.emit_i64(v as i64) }

    fn emit_i8(&mut self, v: i8) -> EncoderResult<()> { self.emit_i64(v as i64) }

    fn emit_i16(&mut self, v: i16) -> EncoderResult<()> { self.emit_i64(v as i64) }

    fn emit_i32(&mut self, v: i32) -> EncoderResult<()> { self.emit_i64(v as i64) }

    fn emit_i64(&mut self, v: i64) -> EncoderResult<()> { expect_value!(self); write!(self.get_writer(), "i{}e", v) }

    fn emit_bool(&mut self, v: bool) -> EncoderResult<()> {
        expect_value!(self);
        if v {
            self.emit_str("true")
        } else {
            self.emit_str("false")
        }
    }

    fn emit_f32(&mut self, v: f32) -> EncoderResult<()> {
        expect_value!(self);
        self.emit_str(&std::f32::to_str_hex(v))
    }

    fn emit_f64(&mut self, v: f64) -> EncoderResult<()> {
        expect_value!(self);
        self.emit_str(&std::f64::to_str_hex(v))
    }

    fn emit_char(&mut self, v: char) -> EncoderResult<()> {
        expect_value!(self);
        self.emit_str(&v.to_string())
    }

    fn emit_str(&mut self, v: &str) -> EncoderResult<()> {
        if self.expect_key {
            self.keys.push(util::ByteString::from_slice(v.as_bytes()));
            Ok(())
        } else {
            try!(write!(self.get_writer(), "{}:", v.len()));
            self.get_writer().write_all(v.as_bytes())
        }
    }

    fn emit_enum<F>(&mut self, _name: &str, _f: F) -> EncoderResult<()> where F: FnOnce(&mut Encoder<'a>) -> EncoderResult<()> {
        self.error("emit_enum not implemented")
    }

    fn emit_enum_variant<F>(&mut self, _v_name: &str, _v_id: usize, _len: usize, _f: F) -> EncoderResult<()> where F: FnOnce(&mut Encoder<'a>) -> EncoderResult<()> {
        self.error("emit_enum_variant not implemented")
    }

    fn emit_enum_variant_arg<F>(&mut self, _a_idx: usize, _f: F) -> EncoderResult<()> where F: FnOnce(&mut Encoder<'a>) -> EncoderResult<()> {
        self.error("emit_enum_variant_arg not implemented")
    }

    fn emit_enum_struct_variant<F>(&mut self, _v_name: &str, _v_id: usize, _len: usize, _f: F) -> EncoderResult<()> where F: FnOnce(&mut Encoder<'a>) -> EncoderResult<()> {
        self.error("emit_enum_struct_variant not implemented")
    }

    fn emit_enum_struct_variant_field<F>(&mut self, _f_name: &str, _f_idx: usize, _f: F) -> EncoderResult<()> where F: FnOnce(&mut Encoder<'a>) -> EncoderResult<()> {
        self.error("emit_enum_struct_variant_field not implemented")
    }

    fn emit_struct<F>(&mut self, _name: &str, _len: usize, f: F) -> EncoderResult<()> where F: FnOnce(&mut Encoder<'a>) -> EncoderResult<()> {
        expect_value!(self);
        self.stack.push(BTreeMap::new());
        try!(f(self));
        let dict = self.stack.pop().unwrap();
        try!(self.encode_dict(&dict));
        self.is_none = false;
        Ok(())
    }

    fn emit_struct_field<F>(&mut self, f_name: &str, _f_idx: usize, f: F) -> EncoderResult<()> where F: FnOnce(&mut Encoder<'a>) -> EncoderResult<()> {
        expect_value!(self);
        self.writers.push(old_io::MemWriter::new());
        try!(f(self));
        let data = self.writers.pop().unwrap();
        let dict = self.stack.last_mut().unwrap();
        if !self.is_none {
            dict.insert(util::ByteString::from_slice(f_name.as_bytes()), data.into_inner());
        }
        self.is_none = false;
        Ok(())
    }

    fn emit_tuple<F>(&mut self, _len: usize, _f: F) -> EncoderResult<()> where F: FnOnce(&mut Encoder<'a>) -> EncoderResult<()> {
        self.error("emit_tuple not implemented")
    }

    fn emit_tuple_arg<F>(&mut self, _idx: usize, _f: F) -> EncoderResult<()> where F: FnOnce(&mut Encoder<'a>) -> EncoderResult<()> {
        self.error("emit_tuple_arg not implemented")
    }
    fn emit_tuple_struct<F>(&mut self, _name: &str, _len: usize, _f: F) -> EncoderResult<()> where F: FnOnce(&mut Encoder<'a>) -> EncoderResult<()> {
        self.error("emit_tuple_struct not implemented")
    }
    fn emit_tuple_struct_arg<F>(&mut self, _f_idx: usize, _f: F) -> EncoderResult<()> where F: FnOnce(&mut Encoder<'a>) -> EncoderResult<()> {
        self.error("emit_tuple_struct_arg not implemented")
    }

    fn emit_option<F>(&mut self, f: F) -> EncoderResult<()> where F: FnOnce(&mut Encoder<'a>) -> EncoderResult<()> {
        expect_value!(self);
        f(self)
    }

    fn emit_option_none(&mut self) -> EncoderResult<()> {
        expect_value!(self);
        self.is_none = true;
        write!(self.get_writer(), "3:nil")
    }

    fn emit_option_some<F>(&mut self, f: F) -> EncoderResult<()> where F: FnOnce(&mut Encoder<'a>) -> EncoderResult<()> {
        expect_value!(self);
        f(self)
    }

    fn emit_seq<F>(&mut self, _len: usize, f: F) -> EncoderResult<()> where F: FnOnce(&mut Encoder<'a>) -> EncoderResult<()> {
        expect_value!(self);
        try!(write!(self.get_writer(), "l"));
        try!(f(self));
        self.is_none = false;
        write!(self.get_writer(), "e")
    }

    fn emit_seq_elt<F>(&mut self, _idx: usize, f: F) -> EncoderResult<()> where F: FnOnce(&mut Encoder<'a>) -> EncoderResult<()> {
        expect_value!(self);
        try!(f(self));
        self.is_none = false;
        Ok(())
    }

    fn emit_map<F>(&mut self, _len: usize, f: F) -> EncoderResult<()> where F: FnOnce(&mut Encoder<'a>) -> EncoderResult<()> {
        expect_value!(self);
        self.stack.push(BTreeMap::new());
        try!(f(self));
        let dict = self.stack.pop().unwrap();
        try!(self.encode_dict(&dict));
        self.is_none = false;
        Ok(())
    }

    fn emit_map_elt_key<F>(&mut self, _idx: usize, f: F) -> EncoderResult<()> where F: FnOnce(&mut Encoder<'a>) -> EncoderResult<()> {
        expect_value!(self);
        self.writers.push(old_io::MemWriter::new());
        self.expect_key = true;
        try!(f(self));
        self.expect_key = false;
        self.is_none = false;
        Ok(())
    }

    fn emit_map_elt_val<F>(&mut self, _idx: usize, f: F) -> EncoderResult<()> where F: FnOnce(&mut Encoder<'a>) -> EncoderResult<()> {
        expect_value!(self);
        try!(f(self));
        let key = self.keys.pop();
        let data = self.writers.pop().unwrap();
        let dict = self.stack.last_mut().unwrap();
        dict.insert(key.unwrap(), data.into_inner());
        self.is_none = false;
        Ok(())
    }
}

pub struct Parser<T> {
    reader: T,
    depth: u32,
}

impl<T: Iterator<Item=BencodeEvent>> Parser<T> {
    pub fn new(reader: T) -> Parser<T> {
        Parser {
            reader: reader,
            depth: 0
        }
    }

    pub fn parse(&mut self) -> Result<Bencode, Error> {
        let next = self.reader.next();
        self.parse_elem(next)
    }

    fn parse_elem(&mut self, current: Option<BencodeEvent>) -> Result<Bencode, Error> {
        let res = match current {
            Some(NumberValue(v)) => Ok(Bencode::Number(v)),
            Some(ByteStringValue(v)) => Ok(Bencode::ByteString(v)),
            Some(ListStart) => self.parse_list(current),
            Some(DictStart) => self.parse_dict(current),
            Some(ParseError(err)) => Err(err),
            None => Ok(Empty),
            x => panic!("[root] Unreachable but got {:?}", x)
        };
        if self.depth == 0 {
            let next = self.reader.next();
            match res {
                Err(_) => res,
                _ => {
                    match next {
                        Some(ParseError(err)) => Err(err),
                        None => res,
                        x => panic!("Unreachable but got {:?}", x)
                    }
                }
            }
        } else {
            res
        }
    }

    fn parse_list(&mut self, mut current: Option<BencodeEvent>) -> Result<Bencode, Error> {
        self.depth += 1;
        let mut list = Vec::new();
        loop {
            current = self.reader.next();
            match current {
                Some(ListEnd) => break,
                Some(ParseError(err)) => return Err(err),
                Some(_) => {
                    match self.parse_elem(current) {
                        Ok(v) => list.push(v),
                        err@Err(_) => return err
                    }
                }
                x => panic!("[list] Unreachable but got {:?}", x)
            }
        }
        self.depth -= 1;
        Ok(Bencode::List(list))
    }

    fn parse_dict(&mut self, mut current: Option<BencodeEvent>) -> Result<Bencode, Error> {
        self.depth += 1;
        let mut map = BTreeMap::new();
        loop {
            current = self.reader.next();
            let key = match current {
                Some(DictEnd) => break,
                Some(DictKey(v)) => util::ByteString::from_vec(v),
                Some(ParseError(err)) => return Err(err),
                x => panic!("[dict] Unreachable but got {:?}", x)
            };
            current = self.reader.next();
            let value = try!(self.parse_elem(current));
            map.insert(key, value);
        }
        self.depth -= 1;
        Ok(Bencode::Dict(map))
    }
}

macro_rules! dec_expect_value(($slf:expr) => {
    if $slf.expect_key {
        return Err(Message("Only 'string' map keys allowed".to_string()))
    }
});

static EMPTY: Bencode = Empty;

#[derive(Eq, PartialEq, Clone, Debug)]
pub enum DecoderError {
    Message(String),
    StringEncoding(Vec<u8>),
    Expecting(&'static str, String),
    Unimplemented(&'static str),
}

pub type DecoderResult<T> = Result<T, DecoderError>;

pub struct Decoder<'a> {
    keys: Vec<util::ByteString>,
    expect_key: bool,
    stack: Vec<&'a Bencode>,
}

impl<'a> Decoder<'a> {
    pub fn new(bencode: &'a Bencode) -> Decoder<'a> {
        Decoder {
            keys: Vec::new(),
            expect_key: false,
            stack: vec![bencode],
        }
    }

    fn try_read<T: FromBencode>(&mut self, ty: &'static str) -> DecoderResult<T> {
        let val = self.stack.pop();
        match val.and_then(|b| FromBencode::from_bencode(b)) {
            Some(v) => Ok(v),
            None => Err(Message(format!("Error decoding value as '{}': {:?}", ty, val)))
        }
    }

    fn unimplemented<T>(&self, m: &'static str) -> DecoderResult<T> {
        Err(Unimplemented(m))
    }
}

impl<'a> serialize::Decoder for Decoder<'a> {
    type Error = DecoderError;

    fn error(&mut self, err: &str) -> DecoderError {
        Message(err.to_string())
    }

    fn read_nil(&mut self) -> DecoderResult<()> {
        dec_expect_value!(self);
        self.try_read("nil")
    }

    fn read_usize(&mut self) -> DecoderResult<usize> {
        dec_expect_value!(self);
        self.try_read("usize")
    }

    fn read_u8(&mut self) -> DecoderResult<u8> {
        dec_expect_value!(self);
        self.try_read("u8")
    }

    fn read_u16(&mut self) -> DecoderResult<u16> {
        dec_expect_value!(self);
        self.try_read("u16")
    }

    fn read_u32(&mut self) -> DecoderResult<u32> {
        dec_expect_value!(self);
        self.try_read("u32")
    }

    fn read_u64(&mut self) -> DecoderResult<u64> {
        dec_expect_value!(self);
        self.try_read("u64")
    }

    fn read_isize(&mut self) -> DecoderResult<isize> {
        dec_expect_value!(self);
        self.try_read("isize")
    }

    fn read_i8(&mut self) -> DecoderResult<i8> {
        dec_expect_value!(self);
        self.try_read("i8")
    }

    fn read_i16(&mut self) -> DecoderResult<i16> {
        dec_expect_value!(self);
        self.try_read("i16")
    }

    fn read_i32(&mut self) -> DecoderResult<i32> {
        dec_expect_value!(self);
        self.try_read("i32")
    }

    fn read_i64(&mut self) -> DecoderResult<i64> {
        dec_expect_value!(self);
        self.try_read("i64")
    }

    fn read_bool(&mut self) -> DecoderResult<bool> {
        dec_expect_value!(self);
        self.try_read("bool")
    }

    fn read_f32(&mut self) -> DecoderResult<f32> {
        dec_expect_value!(self);
        self.try_read("f32")
    }

    fn read_f64(&mut self) -> DecoderResult<f64> {
        dec_expect_value!(self);
        self.try_read("f64")
    }

    fn read_char(&mut self) -> DecoderResult<char> {
        dec_expect_value!(self);
        self.try_read("char")
    }

    fn read_str(&mut self) -> DecoderResult<String> {
        if self.expect_key {
            let b = self.keys.pop().unwrap().unwrap();
            match String::from_utf8(b) {
                Ok(s) => Ok(s),
                Err(err) => Err(StringEncoding(err.into_bytes()))
            }
        } else {
            let bencode = self.stack.pop();
            match bencode {
                Some(&Bencode::ByteString(ref v)) => {
                    String::from_utf8(v.clone()).map_err(|err| StringEncoding(err.into_bytes()))
                }
                _ => Err(self.error(&format!("Error decoding value as str: {:?}", bencode)))
            }
        }
    }

    fn read_enum<T, F>(&mut self, _name: &str, _f: F) -> DecoderResult<T> where F: FnOnce(&mut Decoder<'a>) -> DecoderResult<T> {
        self.unimplemented("read_enum")
    }

    fn read_enum_variant<T, F>(&mut self, _names: &[&str], mut _f: F) -> DecoderResult<T> where F: FnMut(&mut Decoder<'a>, usize) -> DecoderResult<T> {
        self.unimplemented("read_enum_variant")
    }

    fn read_enum_variant_arg<T, F>(&mut self, _idx: usize, _f: F) -> DecoderResult<T> where F: FnOnce(&mut Decoder<'a>) -> DecoderResult<T> {
        self.unimplemented("read_enum_variant_arg")
    }

    fn read_enum_struct_variant<T, F>(&mut self, _names: &[&str], _f: F) -> DecoderResult<T> where F: FnMut(&mut Decoder<'a>, usize) -> DecoderResult<T> {
        self.unimplemented("read_enum_struct_variant")
    }

    fn read_enum_struct_variant_field<T, F>(&mut self, _name: &str, _idx: usize, _f: F) -> DecoderResult<T> where F: FnOnce(&mut Decoder<'a>) -> DecoderResult<T> {
        self.unimplemented("read_enum_struct_variant_field")
    }

    fn read_struct<T, F>(&mut self, _name: &str, _len: usize, f: F) -> DecoderResult<T> where F: FnOnce(&mut Decoder<'a>) -> DecoderResult<T> {
        dec_expect_value!(self);
        let res = try!(f(self));
        self.stack.pop();
        Ok(res)
    }

    fn read_struct_field<T, F>(&mut self, name: &str, _idx: usize, f: F) -> DecoderResult<T> where F: FnOnce(&mut Decoder<'a>) -> DecoderResult<T> {
        dec_expect_value!(self);
        let val = match self.stack.last() {
            Some(v) => {
                match *v {
                    &Bencode::Dict(ref m) => {
                        match m.get(&util::ByteString::from_slice(name.as_bytes())) {
                            Some(v) => v,
                            None => &EMPTY
                        }
                    }
                    _ => return Err(Expecting("Dict", format!("{:?}", v)))
                }
            }
            None => return Err(Expecting("Dict", "None".to_string()))
        };
        self.stack.push(val);
        f(self)
    }

    fn read_tuple<T, F>(&mut self, _tuple_len: usize, _f: F) -> DecoderResult<T> where F: FnOnce(&mut Decoder<'a>) -> DecoderResult<T> {
        self.unimplemented("read_tuple")
    }

    fn read_tuple_arg<T, F>(&mut self, _idx: usize, _f: F) -> DecoderResult<T> where F: FnOnce(&mut Decoder<'a>) -> DecoderResult<T> {
        self.unimplemented("read_tuple_arg")
    }

    fn read_tuple_struct<T, F>(&mut self, _name: &str, _len: usize, _f: F) -> DecoderResult<T> where F: FnOnce(&mut Decoder<'a>) -> DecoderResult<T> {
        self.unimplemented("read_tuple_struct")
    }

    fn read_tuple_struct_arg<T, F>(&mut self, _idx: usize, _f: F) -> DecoderResult<T> where F: FnOnce(&mut Decoder<'a>) -> DecoderResult<T> {
        self.unimplemented("read_tuple_struct_arg")
    }

    fn read_option<T, F>(&mut self, mut f: F) -> DecoderResult<T> where F: FnMut(&mut Decoder<'a>, bool) -> DecoderResult<T> {
        let value = self.stack.pop();
        match value {
            Some(&Bencode::Empty) => f(self, false),
            Some(&Bencode::ByteString(ref v)) => {
                if v == b"nil" {
                    f(self, false)
                } else {
                    self.stack.push(value.unwrap());
                    f(self, true)
                }
            },
            Some(v) => {
                self.stack.push(v);
                f(self, true)
            }
            None => return Err(Expecting("Bencode", "None".to_string()))
        }
    }

    fn read_seq<T, F>(&mut self, f: F) -> DecoderResult<T> where F: FnOnce(&mut Decoder<'a>, usize) -> DecoderResult<T> {
        dec_expect_value!(self);
        let len = match self.stack.pop() {
            Some(&Bencode::List(ref list)) => {
                for v in list.iter().rev() {
                    self.stack.push(v);
                }
                list.len()
            }
            val => return Err(Expecting("List", val.map(|v| v.to_string()).unwrap_or("".to_string())))
        };
        f(self, len)
    }

    fn read_seq_elt<T, F>(&mut self, _idx: usize, f: F) -> DecoderResult<T> where F: FnOnce(&mut Decoder<'a>) -> DecoderResult<T> {
        dec_expect_value!(self);
        f(self)
    }

    fn read_map<T, F>(&mut self, f: F) -> DecoderResult<T> where F: FnOnce(&mut Decoder<'a>, usize) -> DecoderResult<T> {
        dec_expect_value!(self);
        let len = match self.stack.pop() {
            Some(&Bencode::Dict(ref m)) => {
                for (key, value) in m.iter() {
                    self.keys.push(key.clone());
                    self.stack.push(value);
                }
                m.len()
            }
            val => return Err(Expecting("Dict", val.map(|v| v.to_string()).unwrap_or("".to_string())))
        };
        f(self, len)
    }

    fn read_map_elt_key<T, F>(&mut self, _idx: usize, f: F) -> DecoderResult<T> where F: FnOnce(&mut Decoder<'a>) -> DecoderResult<T> {
        dec_expect_value!(self);
        self.expect_key = true;
        let res = try!(f(self));
        self.expect_key = false;
        Ok(res)
    }

    fn read_map_elt_val<T, F>(&mut self, _idx: usize, f: F) -> DecoderResult<T> where F: FnOnce(&mut Decoder<'a>) -> DecoderResult<T> {
        dec_expect_value!(self);
        f(self)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::collections::HashMap;

    use rustc_serialize::{Encodable, Decodable};

    use streaming::Error;
    use streaming::BencodeEvent;
    use streaming::BencodeEvent::{NumberValue, ByteStringValue, ListStart,
                                  ListEnd, DictStart, DictKey, DictEnd, ParseError};

    use super::{Bencode, ToBencode};
    use super::{Parser, Decoder, DecoderResult, encode};

    use super::util;

    macro_rules! assert_encoding(($value:expr, $expected:expr) => ({
        let value = $value;
        let encoded = match encode(&value) {
            Ok(e) => e,
            Err(err) => panic!("Unexpected failure: {}", err)
        };
        assert_eq!($expected, encoded);
    }));

    macro_rules! assert_decoding(($enc:expr, $value:expr) => ({
        let bencode = super::from_vec($enc).unwrap();
        let mut decoder = Decoder::new(&bencode);
        let result = Decodable::decode(&mut decoder);
        assert_eq!(Ok($value), result);
    }));

    macro_rules! gen_encode_test(($name:ident, $(($val:expr) -> $enc:expr),+) => {
        #[test]
        fn $name() {
            $(assert_encoding!($val, $enc);)+
        }
    });

    macro_rules! gen_tobencode_test(($name:ident, $(($val:expr) -> $enc:expr),+) => {
        #[test]
        fn $name() {
            $({
                let value = $val.to_bencode();
                assert_encoding!(value, $enc)
            };)+
        }
    });

    macro_rules! assert_identity(($value:expr) => ({
        let value = $value;
        let encoded = match encode(&value) {
            Ok(e) => e,
            Err(err) => panic!("Unexpected failure: {}", err)
        };
        let bencode = super::from_vec(encoded).unwrap();
        let mut decoder = Decoder::new(&bencode);
        let result = Decodable::decode(&mut decoder);
        assert_eq!(Ok(value), result);
    }));

    macro_rules! gen_identity_test(($name:ident, $($val:expr),+) => {
        #[test]
        fn $name() {
            $(assert_identity!($val);)+
        }
    });

    macro_rules! gen_encode_identity_test(($name_enc:ident, $name_ident:ident, $(($val:expr) -> $enc:expr),+) => {
        gen_encode_test!($name_enc, $(($val) -> $enc),+);
        gen_identity_test!($name_ident, $($val),+);
    });

    macro_rules! gen_complete_test(($name_enc:ident, $name_benc:ident, $name_ident:ident, $(($val:expr) -> $enc:expr),+) => {
        gen_encode_test!($name_enc, $(($val) -> $enc),+);
        gen_tobencode_test!($name_benc, $(($val) -> $enc),+);
        gen_identity_test!($name_ident, $($val),+);
    });

    fn bytes(s: &str) -> Vec<u8> {
        s.as_bytes().to_vec()
    }

    gen_complete_test!(encodes_unit,
                       tobencode_unit,
                       identity_unit,
                       (()) -> bytes("0:"));

    gen_complete_test!(encodes_option_none,
                       tobencode_option_none,
                       identity_option_none,
                       ({
                           let none: Option<isize> = None;
                           none
                       }) -> bytes("3:nil"));

    gen_complete_test!(encodes_option_some,
                       tobencode_option_some,
                       identity_option_some,
                       (Some(1is)) -> bytes("i1e"),
                       (Some("rust".to_string())) -> bytes("4:rust"),
                       (Some(vec![(), ()])) -> bytes("l0:0:e"));

    gen_complete_test!(encodes_nested_option,
                       tobencode_nested_option,
                       identity_nested_option,
                       (Some(Some(1is))) -> bytes("i1e"),
                       (Some(Some("rust".to_string()))) -> bytes("4:rust"));

    #[test]
    fn option_is_none_if_any_nested_option_is_none() {
        let value: Option<Option<isize>> = Some(None);
        let encoded = match encode(&value) {
            Ok(e) => e,
            Err(err) => panic!("Unexpected failure: {}", err)
        };
        let none: Option<Option<isize>> = None;
        assert_decoding!(encoded, none);
    }

    gen_complete_test!(encodes_zero_isize,
                       tobencode_zero_isize,
                       identity_zero_isize,
                       (0is) -> bytes("i0e"));

    gen_complete_test!(encodes_positive_isize,
                       tobencode_positive_isize,
                       identity_positive_isize,
                       (5is) -> bytes("i5e"),
                       (99is) -> bytes("i99e"),
                       (::std::isize::MAX) -> bytes(format!("i{}e", ::std::isize::MAX).as_slice()));

    gen_complete_test!(encodes_negative_isize,
                       tobencode_negative_isize,
                       identity_negative_isize,
                       (-5is) -> bytes("i-5e"),
                       (-99is) -> bytes("i-99e"),
                       (::std::isize::MIN) -> bytes(format!("i{}e", ::std::isize::MIN).as_slice()));

    gen_complete_test!(encodes_zero_i8,
                       tobencode_zero_i8,
                       identity_zero_i8,
                       (0i8) -> bytes("i0e"));

    gen_complete_test!(encodes_positive_i8,
                       tobencode_positive_i8,
                       identity_positive_i8,
                       (5i8) -> bytes("i5e"),
                       (99i8) -> bytes("i99e"),
                       (::std::i8::MAX) -> bytes(format!("i{}e", ::std::i8::MAX).as_slice()));

    gen_complete_test!(encodes_negative_i8,
                       tobencode_negative_i8,
                       identity_negative_i8,
                       (-5i8) -> bytes("i-5e"),
                       (-99i8) -> bytes("i-99e"),
                       (::std::i8::MIN) -> bytes(format!("i{}e", ::std::i8::MIN).as_slice()));

    gen_complete_test!(encodes_zero_i16,
                       tobencode_zero_i16,
                       identity_zero_i16,
                       (0i16) -> bytes("i0e"));

    gen_complete_test!(encodes_positive_i16,
                       tobencode_positive_i16,
                       identity_positive_i16,
                       (5i16) -> bytes("i5e"),
                       (99i16) -> bytes("i99e"),
                       (::std::i16::MAX) -> bytes(format!("i{}e", ::std::i16::MAX).as_slice()));

    gen_complete_test!(encodes_negative_i16,
                       tobencode_negative_i16,
                       identity_negative_i16,
                       (-5i16) -> bytes("i-5e"),
                       (-99i16) -> bytes("i-99e"),
                       (::std::i16::MIN) -> bytes(format!("i{}e", ::std::i16::MIN).as_slice()));

    gen_complete_test!(encodes_zero_i32,
                       tobencode_zero_i32,
                       identity_zero_i32,
                       (0i32) -> bytes("i0e"));

    gen_complete_test!(encodes_positive_i32,
                       tobencode_positive_i32,
                       identity_positive_i32,
                       (5i32) -> bytes("i5e"),
                       (99i32) -> bytes("i99e"),
                       (::std::i32::MAX) -> bytes(format!("i{}e", ::std::i32::MAX).as_slice()));

    gen_complete_test!(encodes_negative_i32,
                       tobencode_negative_i32,
                       identity_negative_i32,
                       (-5i32) -> bytes("i-5e"),
                       (-99i32) -> bytes("i-99e"),
                       (::std::i32::MIN) -> bytes(format!("i{}e", ::std::i32::MIN).as_slice()));

    gen_complete_test!(encodes_zero_i64,
                       tobencode_zero_i64,
                       identity_zero_i64,
                       (0i64) -> bytes("i0e"));

    gen_complete_test!(encodes_positive_i64,
                       tobencode_positive_i64,
                       identity_positive_i64,
                       (5i64) -> bytes("i5e"),
                       (99i64) -> bytes("i99e"),
                       (::std::i64::MAX) -> bytes(format!("i{}e", ::std::i64::MAX).as_slice()));

    gen_complete_test!(encodes_negative_i64,
                       tobencode_negative_i64,
                       identity_negative_i64,
                       (-5i64) -> bytes("i-5e"),
                       (-99i64) -> bytes("i-99e"),
                       (::std::i64::MIN) -> bytes(format!("i{}e", ::std::i64::MIN).as_slice()));

    gen_complete_test!(encodes_zero_usize,
                       tobencode_zero_usize,
                       identity_zero_usize,
                       (0us) -> bytes("i0e"));

    gen_complete_test!(encodes_positive_usize,
                       tobencode_positive_usize,
                       identity_positive_usize,
                       (5us) -> bytes("i5e"),
                       (99us) -> bytes("i99e"),
                       (::std::usize::MAX / 2) -> bytes(format!("i{}e", ::std::usize::MAX / 2).as_slice()));

    gen_complete_test!(encodes_zero_u8,
                       tobencode_zero_u8,
                       identity_zero_u8,
                       (0u8) -> bytes("i0e"));

    gen_complete_test!(encodes_positive_u8,
                       tobencode_positive_u8,
                       identity_positive_u8,
                       (5u8) -> bytes("i5e"),
                       (99u8) -> bytes("i99e"),
                       (::std::u8::MAX) -> bytes(format!("i{}e", ::std::u8::MAX).as_slice()));

    gen_complete_test!(encodes_zero_u16,
                       tobencode_zero_u16,
                       identity_zero_u16,
                       (0u16) -> bytes("i0e"));

    gen_complete_test!(encodes_positive_u16,
                       tobencode_positive_u16,
                       identity_positive_u16,
                       (5u16) -> bytes("i5e"),
                       (99u16) -> bytes("i99e"),
                       (::std::u16::MAX) -> bytes(format!("i{}e", ::std::u16::MAX).as_slice()));

    gen_complete_test!(encodes_zero_u32,
                       tobencode_zero_u32,
                       identity_zero_u32,
                       (0u32) -> bytes("i0e"));

    gen_complete_test!(encodes_positive_u32,
                       tobencode_positive_u32,
                       identity_positive_u32,
                       (5u32) -> bytes("i5e"),
                       (99u32) -> bytes("i99e"),
                       (::std::u32::MAX) -> bytes(format!("i{}e", ::std::u32::MAX).as_slice()));

    gen_complete_test!(encodes_zero_u64,
                       tobencode_zero_u64,
                       identity_zero_u64,
                       (0u64) -> bytes("i0e"));

    gen_complete_test!(encodes_positive_u64,
                       tobencode_positive_u64,
                       identity_positive_u64,
                       (5u64) -> bytes("i5e"),
                       (99u64) -> bytes("i99e"),
                       (::std::u64::MAX / 2) -> bytes(format!("i{}e", ::std::u64::MAX / 2).as_slice()));

    gen_complete_test!(encodes_bool,
                       tobencode_bool,
                       identity_bool,
                       (true) -> bytes("4:true"),
                       (false) -> bytes("5:false"));

    gen_complete_test!(encodes_zero_f32,
                       tobencode_zero_f32,
                       identity_zero_f32,
                       (0.0f32) -> bytes("1:0"));

    gen_complete_test!(encodes_positive_f32,
                       tobencode_positive_f32,
                       identity_positive_f32,
                       (99.0f32) -> bytes("2:63"),
                       (101.12345f32) -> bytes("8:65.1f9a8"));

    gen_complete_test!(encodes_negative_f32,
                       tobencode_negative_f32,
                       identity_negative_f32,
                       (-99.0f32) -> bytes("3:-63"),
                       (-101.12345f32) -> bytes("9:-65.1f9a8"));

    gen_complete_test!(encodes_zero_f64,
                       tobencode_zero_f64,
                       identity_zero_f64,
                       (0.0f64) -> bytes("1:0"));

    gen_complete_test!(encodes_positive_f64,
                       tobencode_positive_f64,
                       identity_positive_f64,
                       (99.0f64) -> bytes("2:63"),
                       (101.12345f64) -> bytes("15:65.1f9a6b50b0f4"));

    gen_complete_test!(encodes_negative_f64,
                       tobencode_negative_f64,
                       identity_negative_f64,
                       (-99.0f64) -> bytes("3:-63"),
                       (-101.12345f64) -> bytes("16:-65.1f9a6b50b0f4"));

    gen_complete_test!(encodes_lower_letter_char,
                       tobencode_lower_letter_char,
                       identity_lower_letter_char,
                       ('a') -> bytes("1:a"),
                       ('c') -> bytes("1:c"),
                       ('z') -> bytes("1:z"));

    gen_complete_test!(encodes_upper_letter_char,
                       tobencode_upper_letter_char,
                       identity_upper_letter_char,
                       ('A') -> bytes("1:A"),
                       ('C') -> bytes("1:C"),
                       ('Z') -> bytes("1:Z"));

    gen_complete_test!(encodes_multibyte_char,
                       tobencode_multibyte_char,
                       identity_multibyte_char,
                       ('ệ') -> bytes("3:ệ"),
                       ('虎') -> bytes("3:虎"));

    gen_complete_test!(encodes_control_char,
                       tobencode_control_char,
                       identity_control_char,
                       ('\n') -> bytes("1:\n"),
                       ('\r') -> bytes("1:\r"),
                       ('\0') -> bytes("1:\0"));

    gen_complete_test!(encode_empty_str,
                      tobencode_empty_str,
                      identity_empty_str,
                      ("".to_string()) -> bytes("0:"));

    gen_complete_test!(encode_str,
                      tobencode_str,
                      identity_str,
                      ("a".to_string()) -> bytes("1:a"),
                      ("foo".to_string()) -> bytes("3:foo"),
                      ("This is nice!?#$%".to_string()) -> bytes("17:This is nice!?#$%"));

    gen_complete_test!(encode_str_with_multibyte_chars,
                      tobencode_str_with_multibyte_chars,
                      identity_str_with_multibyte_chars,
                      ("Löwe 老虎 Léopard".to_string()) -> bytes("21:Löwe 老虎 Léopard"),
                      ("いろはにほへとちりぬるを".to_string()) -> bytes("36:いろはにほへとちりぬるを"));

    gen_complete_test!(encodes_empty_vec,
                       tobencode_empty_vec,
                       identity_empty_vec,
                       ({
                           let empty: Vec<u8> = Vec::new();
                           empty
                       }) -> bytes("le"));

    gen_complete_test!(encodes_nonmpty_vec,
                       tobencode_nonmpty_vec,
                       identity_nonmpty_vec,
                       (vec![0is, 1is, 3is, 4is]) -> bytes("li0ei1ei3ei4ee"),
                       (vec!["foo".to_string(), "b".to_string()]) -> bytes("l3:foo1:be"));

    gen_complete_test!(encodes_nested_vec,
                       tobencode_nested_vec,
                       identity_nested_vec,
                       (vec![vec![1is], vec![2is, 3is], vec![]]) -> bytes("lli1eeli2ei3eelee"));

    #[derive(Eq, PartialEq, Debug, RustcEncodable, RustcDecodable)]
    struct SimpleStruct {
        a: usize,
        b: Vec<String>,
    }

    #[derive(Eq, PartialEq, Debug, RustcEncodable, RustcDecodable)]
    struct InnerStruct {
        field_one: (),
        list: Vec<usize>,
        abc: String
    }

    #[derive(Eq, PartialEq, Debug, RustcEncodable, RustcDecodable)]
    struct OuterStruct {
        inner: Vec<InnerStruct>,
        is_true: bool
    }

    gen_encode_identity_test!(encodes_struct,
                              identity_struct,
                              (SimpleStruct {
                                  b: vec!["foo".to_string(), "baar".to_string()],
                                  a: 123
                              }) -> bytes("d1:ai123e1:bl3:foo4:baaree"),
                              (SimpleStruct {
                                  a: 1234567890,
                                  b: vec![]
                              }) -> bytes("d1:ai1234567890e1:blee"));

    gen_encode_identity_test!(encodes_nested_struct,
                              identity_nested_struct,
                              (OuterStruct {
                                  is_true: true,
                                  inner: vec![InnerStruct {
                                      field_one: (),
                                      list: vec![99us, 5us],
                                      abc: "rust".to_string()
                                  }, InnerStruct {
                                      field_one: (),
                                      list: vec![],
                                      abc: "".to_string()
                                  }]
                              }) -> bytes("d\
                                           5:inner\
                                             l\
                                               d\
                                                 3:abc4:rust\
                                                 9:field_one0:\
                                                 4:list\
                                                   l\
                                                     i99e\
                                                     i5e\
                                                   e\
                                               e\
                                               d\
                                                 3:abc0:\
                                                 9:field_one0:\
                                                 4:listle\
                                               e\
                                             e\
                                           7:is_true4:true\
                                          e"));

    macro_rules! map(($m:ident, $(($key:expr, $val:expr)),*) => {{
        let mut _m = $m::new();
        $(_m.insert($key, $val);)*
        _m
    }});

    gen_complete_test!(encodes_hashmap,
                       bencode_hashmap,
                       identity_hashmap,
                       (map!(HashMap, ("a".to_string(), 1is))) -> bytes("d1:ai1ee"),
                       (map!(HashMap, ("foo".to_string(), "a".to_string()), ("bar".to_string(), "bb".to_string()))) -> bytes("d3:bar2:bb3:foo1:ae"));

    gen_complete_test!(encodes_nested_hashmap,
                       bencode_nested_hashmap,
                       identity_nested_hashmap,
                       (map!(HashMap, ("a".to_string(), map!(HashMap, ("foo".to_string(), 101is), ("bar".to_string(), 102is))))) -> bytes("d1:ad3:bari102e3:fooi101eee"));
    #[test]
    fn decode_error_on_wrong_map_key_type() {
        let benc = Bencode::Dict(map!(BTreeMap, (util::ByteString::from_vec(bytes("foo")), Bencode::ByteString(bytes("bar")))));
        let mut decoder = Decoder::new(&benc);
        let res: DecoderResult<BTreeMap<isize, String>> = Decodable::decode(&mut decoder);
        assert!(res.is_err());
    }

    #[test]
    fn encode_error_on_wrong_map_key_type() {
        let m = map!(HashMap, (1is, "foo"));
        let encoded = encode(&m);
        assert!(encoded.is_err())
    }

    #[test]
    fn encodes_struct_fields_in_sorted_order() {
        #[derive(RustcEncodable)]
        struct OrderedStruct {
            z: isize,
            a: isize,
            ab: isize,
            aa: isize,
        }
        let s = OrderedStruct {
            z: 4,
            a: 1,
            ab: 3,
            aa: 2
        };
        assert_eq!(encode(&s), Ok(bytes("d1:ai1e2:aai2e2:abi3e1:zi4ee")));
    }

    #[derive(RustcEncodable, RustcDecodable, Eq, PartialEq, Debug, Clone)]
    struct OptionalStruct {
        a: Option<isize>,
        b: isize,
        c: Option<Vec<Option<bool>>>,
    }

    #[derive(RustcEncodable, RustcDecodable, Eq, PartialEq, Debug)]
    struct OptionalStructOuter {
        a: Option<OptionalStruct>,
        b: Option<isize>,
    }

    static OPT_STRUCT: OptionalStruct = OptionalStruct {
        a: None,
        b: 10,
        c: None
    };

    #[test]
    fn struct_option_none_fields_are_not_encoded() {
        assert_encoding!(OPT_STRUCT.clone(), bytes("d1:bi10ee"));
    }


    #[test]
    fn struct_options_not_present_default_to_none() {
        assert_decoding!(bytes("d1:bi10ee"), OPT_STRUCT.clone());
    }

    gen_encode_identity_test!(encodes_nested_struct_fields,
                              identity_nested_struct_field,
                              ({
                                  OptionalStructOuter {
                                      a: Some(OPT_STRUCT.clone()),
                                      b: None
                                  }
                              }) -> bytes("d1:ad1:bi10eee"),
                              ({
                                  let a = OptionalStruct {
                                      a: None,
                                      b: 10,
                                      c: Some(vec![Some(true), None])
                                  };
                                  OptionalStructOuter {
                                      a: Some(a),
                                      b: Some(99)
                                  }
                              }) -> bytes("d1:ad1:bi10e1:cl4:true3:nilee1:bi99ee"));

    fn try_bencode(bencode: Bencode) -> Vec<u8> {
        match bencode.to_bytes() {
            Ok(v) => v,
            Err(err) => panic!("Unexpected error: {}", err)
        }
    }

    #[test]
    fn encodes_empty_bytestring() {
        assert_eq!(try_bencode(Bencode::ByteString(Vec::new())), bytes("0:"));
    }

    #[test]
    fn encodes_nonempty_bytestring() {
        assert_eq!(try_bencode(Bencode::ByteString(b"abc".to_vec())), bytes("3:abc"));
        assert_eq!(try_bencode(Bencode::ByteString(vec![0, 1, 2, 3])), bytes("4:\x00\x01\x02\x03"));
    }

    #[test]
    fn encodes_empty_list() {
        assert_eq!(try_bencode(Bencode::List(Vec::new())), bytes("le"));
    }

    #[test]
    fn encodes_nonempty_list() {
        assert_eq!(try_bencode(Bencode::List(vec![Bencode::Number(1)])), bytes("li1ee"));
        assert_eq!(try_bencode(Bencode::List(vec![Bencode::ByteString("foobar".as_bytes().to_vec()),
                          Bencode::Number(-1)])), bytes("l6:foobari-1ee"));
    }

    #[test]
    fn encodes_nested_list() {
        assert_eq!(try_bencode(Bencode::List(vec![Bencode::List(vec![])])), bytes("llee"));
        let list = Bencode::List(vec![Bencode::Number(1988), Bencode::List(vec![Bencode::Number(2014)])]);
        assert_eq!(try_bencode(list), bytes("li1988eli2014eee"));
    }

    #[test]
    fn encodes_empty_dict() {
        assert_eq!(try_bencode(Bencode::Dict(BTreeMap::new())), bytes("de"));
    }

    #[test]
    fn encodes_dict_with_items() {
        let mut m = BTreeMap::new();
        m.insert(util::ByteString::from_str("k1"), Bencode::Number(1));
        assert_eq!(try_bencode(Bencode::Dict(m.clone())), bytes("d2:k1i1ee"));
        m.insert(util::ByteString::from_str("k2"), Bencode::ByteString(vec![0, 0]));
        assert_eq!(try_bencode(Bencode::Dict(m.clone())), bytes("d2:k1i1e2:k22:\0\0e"));
    }

    #[test]
    fn encodes_nested_dict() {
        let mut outer = BTreeMap::new();
        let mut inner = BTreeMap::new();
        inner.insert(util::ByteString::from_str("val"), Bencode::ByteString(vec![68, 0, 90]));
        outer.insert(util::ByteString::from_str("inner"), Bencode::Dict(inner));
        assert_eq!(try_bencode(Bencode::Dict(outer)), bytes("d5:innerd3:val3:D\0Zee"));
    }

    #[test]
    fn encodes_dict_fields_in_sorted_order() {
        let mut m = BTreeMap::new();
        m.insert(util::ByteString::from_str("z"), Bencode::Number(1));
        m.insert(util::ByteString::from_str("abd"), Bencode::Number(3));
        m.insert(util::ByteString::from_str("abc"), Bencode::Number(2));
        assert_eq!(try_bencode(Bencode::Dict(m)), bytes("d3:abci2e3:abdi3e1:zi1ee"));
    }

    fn assert_decoded_eq(events: &[BencodeEvent], expected: Result<Bencode, Error>) {
        let mut parser = Parser::new(events.to_vec().into_iter());
        let result = parser.parse();
        assert_eq!(expected, result);
    }

    #[test]
    fn decodes_empty_input() {
        assert_decoded_eq(&[], Ok(Bencode::Empty));
    }

    #[test]
    fn decodes_number() {
        assert_decoded_eq(&[NumberValue(25)], Ok(Bencode::Number(25)));
    }

    #[test]
    fn decodes_bytestring() {
        assert_decoded_eq(&[ByteStringValue(bytes("foo"))], Ok(Bencode::ByteString(bytes("foo"))));
    }

    #[test]
    fn decodes_empty_list() {
        assert_decoded_eq(&[ListStart, ListEnd], Ok(Bencode::List(vec![])));
    }

    #[test]
    fn decodes_list_with_elements() {
        assert_decoded_eq(&[ListStart,
                            NumberValue(1),
                            ListEnd], Ok(Bencode::List(vec![Bencode::Number(1)])));
        assert_decoded_eq(&[ListStart,
                            ByteStringValue(bytes("str")),
                            NumberValue(11),
                            ListEnd], Ok(Bencode::List(vec![Bencode::ByteString(bytes("str")),
                                               Bencode::Number(11)])));
    }

    #[test]
    fn decodes_nested_list() {
        assert_decoded_eq(&[ListStart,
                            ListStart,
                            NumberValue(13),
                            ListEnd,
                            ByteStringValue(bytes("rust")),
                            ListEnd],
                            Ok(Bencode::List(vec![Bencode::List(vec![Bencode::Number(13)]),
                                      Bencode::ByteString(bytes("rust"))])));
    }

    #[test]
    fn decodes_empty_dict() {
        assert_decoded_eq(&[DictStart, DictEnd], Ok(Bencode::Dict(BTreeMap::new())));
    }

    #[test]
    fn decodes_dict_with_value() {
        let mut map = BTreeMap::new();
        map.insert(util::ByteString::from_str("foo"), Bencode::ByteString(bytes("rust")));
        assert_decoded_eq(&[DictStart,
                            DictKey(bytes("foo")),
                            ByteStringValue(bytes("rust")),
                            DictEnd], Ok(Bencode::Dict(map)));
    }

    #[test]
    fn decodes_dict_with_values() {
        let mut map = BTreeMap::new();
        map.insert(util::ByteString::from_str("num"), Bencode::Number(9));
        map.insert(util::ByteString::from_str("str"), Bencode::ByteString(bytes("abc")));
        map.insert(util::ByteString::from_str("list"), Bencode::List(vec![Bencode::Number(99)]));
        assert_decoded_eq(&[DictStart,
                            DictKey(bytes("num")),
                            NumberValue(9),
                            DictKey(bytes("str")),
                            ByteStringValue(bytes("abc")),
                            DictKey(bytes("list")),
                            ListStart,
                            NumberValue(99),
                            ListEnd,
                            DictEnd], Ok(Bencode::Dict(map)));
    }

    #[test]
    fn decodes_nested_dict() {
        let mut inner = BTreeMap::new();
        inner.insert(util::ByteString::from_str("inner"), Bencode::Number(2));
        let mut outer = BTreeMap::new();
        outer.insert(util::ByteString::from_str("dict"), Bencode::Dict(inner));
        outer.insert(util::ByteString::from_str("outer"), Bencode::Number(1));
        assert_decoded_eq(&[DictStart,
                            DictKey(bytes("outer")),
                            NumberValue(1),
                            DictKey(bytes("dict")),
                            DictStart,
                            DictKey(bytes("inner")),
                            NumberValue(2),
                            DictEnd,
                            DictEnd], Ok(Bencode::Dict(outer)));
    }

    #[test]
    fn decode_error_on_parse_error() {
        let err = Error{ pos: 1, msg: "error msg".to_string() };
        let perr = ParseError(err.clone());
        assert_decoded_eq(&[perr.clone()], Err(err.clone()));
        assert_decoded_eq(&[NumberValue(1), perr.clone()], Err(err.clone()));
        assert_decoded_eq(&[ListStart,
                           perr.clone()], Err(err.clone()));
        assert_decoded_eq(&[ListStart,
                           ByteStringValue(bytes("foo")),
                           perr.clone()], Err(err.clone()));
        assert_decoded_eq(&[DictStart,
                            perr.clone()], Err(err.clone()));
        assert_decoded_eq(&[DictStart,
                            DictKey(bytes("foo")),
                           perr.clone()], Err(err.clone()));
    }
}

#[cfg(test)]
mod bench {
    extern crate test;

    use self::test::Bencher;

    use std::old_io;

    use rustc_serialize::{Encodable, Decodable};

    use streaming::StreamingParser;
    use super::{Encoder, Decoder, Parser, DecoderResult, encode};

    #[bench]
    fn encode_large_vec_of_usize(bh: &mut Bencher) {
        let v: Vec<u32> = (0u32..100).collect();
        bh.iter(|| {
            let mut w = old_io::MemWriter::with_capacity(v.len() * 10);
            {
                let mut enc = Encoder::new(&mut w);
                let _ = v.encode(&mut enc);
            }
            w.into_inner()
        });
        bh.bytes = v.len() as u64 * 4;
    }


    #[bench]
    fn decode_large_vec_of_usize(bh: &mut Bencher) {
        let v: Vec<u32> = (0u32..100).collect();
        let b = encode(&v).unwrap();
        bh.iter(|| {
            let streaming_parser = StreamingParser::new(b.clone().into_iter());
            let mut parser = Parser::new(streaming_parser);
            let bencode = parser.parse().unwrap();
            let mut decoder = Decoder::new(&bencode);
            let result: DecoderResult<Vec<usize>> = Decodable::decode(&mut decoder);
            result
        });
        bh.bytes = b.len() as u64 * 4;
    }
}

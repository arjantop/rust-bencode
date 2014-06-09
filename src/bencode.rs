// Copyright 2014 Arjan Topolovec
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_id = "bencode"]
#![license = "MIT/ASL2"]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![feature(macro_rules)]

/*!
  Bencode parsing and serialization

  # Encoding

  ## Using `Encodable`

  ```rust
  extern crate serialize;
  extern crate bencode;

  use serialize::Encodable;

  use bencode::Encoder;

  #[deriving(Encodable)]
  struct MyStruct {
      string: String,
      id: uint,
  }

  fn main() {
      let s = MyStruct { string: "Hello bencode".to_string(), id: 1 };
      let result: Vec<u8> = Encoder::buffer_encode(&s).unwrap();
  }
  ```

  ## Using `ToBencode`

  ```rust
  extern crate collections;
  extern crate bencode;

  use collections::TreeMap;

  use bencode::{Key, ToBencode};

  struct MyStruct {
      a: int,
      b: String,
      c: ~[u8],
  }

  impl ToBencode for MyStruct {
      fn to_bencode(&self) -> bencode::Bencode {
          let mut m = TreeMap::new();
          m.insert(Key::from_str("a"), self.a.to_bencode());
          m.insert(Key::from_str("b"), self.b.to_bencode());
          m.insert(Key::from_str("c"), bencode::ByteString(Vec::from_slice(self.c.as_slice())));
          bencode::Dict(m)
      }
  }

  fn main() {
      let s = MyStruct{ a: 5, b: "foo".to_string(), c: ~[1, 2, 3, 4] };
      let bencode: bencode::Bencode = s.to_bencode();
      let result: Vec<u8> = bencode.to_bytes().unwrap();
  }

  ```

  # Decoding

  ## Using `Decodable`

  ```rust
  extern crate serialize;
  extern crate bencode;

  use serialize::{Encodable, Decodable};

  use bencode::{Encoder, Decoder};

  #[deriving(Encodable, Decodable, PartialEq)]
  struct MyStruct {
      a: int,
      b: String,
      c: ~[u8],
  }

  fn main() {
      let s = MyStruct{ a: 5, b: "foo".to_string(), c: ~[1, 2, 3, 4] };
      let enc: Vec<u8> = Encoder::buffer_encode(&s).unwrap();

      let bencode: bencode::Bencode = bencode::from_vec(enc).unwrap();
      let mut decoder = Decoder::new(&bencode);
      let result: MyStruct = Decodable::decode(&mut decoder).unwrap();
      assert!(s == result)
  }
  ```

  ## Using `FromBencode`

  ```rust
  extern crate collections;
  extern crate bencode;

  use collections::TreeMap;

  use bencode::{FromBencode, ToBencode, Dict, Key};

  #[deriving(PartialEq)]
  struct MyStruct {
      a: int
  }

  impl ToBencode for MyStruct {
      fn to_bencode(&self) -> bencode::Bencode {
          let mut m = TreeMap::new();
          m.insert(Key::from_str("a"), self.a.to_bencode());
          bencode::Dict(m)
      }
  }

  impl FromBencode for MyStruct {
      fn from_bencode(bencode: &bencode::Bencode) -> Option<MyStruct> {
          match bencode {
              &Dict(ref m) => {
                  match m.find(&Key::from_str("a")) {
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
  extern crate serialize;
  extern crate bencode;

  use bencode::streaming;
  use bencode::streaming::StreamingParser;
  use serialize::Encodable;

  use bencode::Encoder;

  #[deriving(Encodable, Decodable, PartialEq)]
  struct MyStruct {
      a: int,
      b: String,
      c: ~[u8],
  }

  fn main() {
      let s = MyStruct{ a: 5, b: "foo".to_string(), c: ~[1, 2, 3, 4] };
      let enc: Vec<u8> = Encoder::buffer_encode(&s).unwrap();

      let mut streaming = StreamingParser::new(enc.move_iter());
      for event in streaming {
          match event {
              streaming::DictStart => println!("dict start"),
              streaming::DictEnd => println!("dict end"),
              streaming::NumberValue(n) => println!("number = {}", n),
              // ...
              _ => println!("Unhandled event: {}", event)
          }
      }
  }
  ```
*/

extern crate serialize;

use std::io;
use std::io::{IoResult, IoError};
use std::fmt;
use std::str;
use std::str::raw;
use std::vec::Vec;

use serialize::{Encodable};

use std::collections::treemap::TreeMap;
use std::collections::hashmap::HashMap;

use streaming::{StreamingParser, Error};
use streaming::{BencodeEvent, NumberValue, ByteStringValue, ListStart,
                ListEnd, DictStart, DictKey, DictEnd, ParseError};

pub mod streaming;

#[deriving(PartialEq, Clone)]
pub enum Bencode {
    Empty,
    Number(i64),
    ByteString(Vec<u8>),
    List(List),
    Dict(Dict),
}

impl fmt::Show for Bencode {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &Empty => { Ok(()) }
            &Number(v) => write!(fmt, "{}", v),
            &ByteString(ref v) => write!(fmt, "s{}", v),
            &List(ref v) => write!(fmt, "{}", v),
            &Dict(ref v) => {
                try!(write!(fmt, r"\{"));
                let mut first = true;
                for (key, value) in v.iter() {
                    if first {
                        first = false;
                    } else {
                        try!(write!(fmt, ", "));
                    }
                    try!(write!(fmt, "{}: {}", *key, *value));
                }
                write!(fmt, r"\}")
            }
        }
    }
}

#[deriving(Eq, PartialEq, Clone, Ord, PartialOrd, Show, Hash)]
pub struct Key(Vec<u8>);

pub type List = Vec<Bencode>;
pub type Dict = TreeMap<Key, Bencode>;

impl Bencode {
    pub fn to_writer(&self, writer: &mut io::Writer) -> io::IoResult<()> {
        let mut encoder = Encoder::new(writer);
        self.encode(&mut encoder)
    }

    pub fn to_bytes(&self) -> io::IoResult<Vec<u8>> {
        let mut writer = io::MemWriter::new();
        match self.to_writer(&mut writer) {
            Ok(_) => Ok(writer.unwrap()),
            Err(err) => Err(err)
        }
    }
}

impl Key {
    pub fn from_str(s: &'static str) -> Key {
        Key(Vec::from_slice(s.as_bytes()))
    }
}

impl<E, S: serialize::Encoder<E>> Encodable<S, E> for Bencode {
    fn encode(&self, e: &mut S) -> Result<(), E> {
        match self {
            &Empty => Ok(()),
            &Number(v) => e.emit_i64(v),
            &ByteString(ref v) => e.emit_str(unsafe { raw::from_utf8(v.as_slice()) }),
            &List(ref v) => v.encode(e),
            &Dict(ref v) => v.encode(e)
        }
    }
}

impl<E, S: serialize::Encoder<E>> Encodable<S, E> for Key {
    fn encode(&self, e: &mut S) -> Result<(), E> {
        let &Key(ref key) = self;
        e.emit_str(unsafe { raw::from_utf8(key.as_slice()) })
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
        ByteString(Vec::new())
    }
}

impl FromBencode for () {
    fn from_bencode(bencode: &Bencode) -> Option<()> {
        match bencode {
            &ByteString(ref v) => {
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
            &None => ByteString(Vec::from_slice(bytes!("nil")))
        }
    }
}

impl<T: FromBencode> FromBencode for Option<T> {
    fn from_bencode(bencode: &Bencode) -> Option<Option<T>> {
        match bencode {
            &ByteString(ref v) => {
                if v.as_slice() == bytes!("nil") {
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
        fn to_bencode(&self) -> Bencode { Number(*self as i64) }
    }
))

macro_rules! derive_num_from_bencode(($t:ty) => (
    impl FromBencode for $t {
        fn from_bencode(bencode: &Bencode) -> Option<$t> {
            match bencode {
                &Number(v) => Some(v as $t),
                _ => None
            }
        }
    }
))

derive_num_to_bencode!(int)
derive_num_from_bencode!(int)

derive_num_to_bencode!(i8)
derive_num_from_bencode!(i8)

derive_num_to_bencode!(i16)
derive_num_from_bencode!(i16)

derive_num_to_bencode!(i32)
derive_num_from_bencode!(i32)

derive_num_to_bencode!(i64)
derive_num_from_bencode!(i64)

derive_num_to_bencode!(uint)
derive_num_from_bencode!(uint)

derive_num_to_bencode!(u8)
derive_num_from_bencode!(u8)

derive_num_to_bencode!(u16)
derive_num_from_bencode!(u16)

derive_num_to_bencode!(u32)
derive_num_from_bencode!(u32)

derive_num_to_bencode!(u64)
derive_num_from_bencode!(u64)

impl ToBencode for f32 {
    fn to_bencode(&self) -> Bencode {
        ByteString(Vec::from_slice(std::f32::to_str_hex(*self).as_bytes()))
    }
}

impl FromBencode for f32 {
    fn from_bencode(bencode: &Bencode) -> Option<f32> {
        match bencode {
            &ByteString(ref v)  => {
                match str::from_utf8(v.as_slice()) {
                    Some(s) => std::f32::from_str_hex(s),
                    None => None
                }
            }
            _ => None
        }
    }
}

impl ToBencode for f64 {
    fn to_bencode(&self) -> Bencode {
        ByteString(Vec::from_slice(std::f64::to_str_hex(*self).as_bytes()))
    }
}

impl FromBencode for f64 {
    fn from_bencode(bencode: &Bencode) -> Option<f64> {
        match bencode {
            &ByteString(ref v)  => {
                match str::from_utf8(v.as_slice()) {
                    Some(s) => std::f64::from_str_hex(s),
                    None => None
                }
            }
            _ => None
        }
    }
}

impl ToBencode for bool {
    fn to_bencode(&self) -> Bencode {
        if *self {
            ByteString(Vec::from_slice(bytes!("true")))
        } else {
            ByteString(Vec::from_slice(bytes!("false")))
        }
    }
}

impl FromBencode for bool {
    fn from_bencode(bencode: &Bencode) -> Option<bool> {
        match bencode {
            &ByteString(ref v) => {
                if v.as_slice() == bytes!("true") {
                    Some(true)
                } else if v.as_slice() == bytes!("false") {
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
        ByteString(Vec::from_slice(self.to_str().as_bytes()))
    }
}

impl FromBencode for char {
    fn from_bencode(bencode: &Bencode) -> Option<char> {
        let s: Option<String> = FromBencode::from_bencode(bencode);
        s.and_then(|s| {
            if s.as_slice().char_len() == 1 {
                Some(s.as_slice().char_at(0))
            } else {
                None
            }
        })
    }
}

impl ToBencode for String {
    fn to_bencode(&self) -> Bencode { ByteString(Vec::from_slice(self.as_bytes())) }
}

impl FromBencode for String {
    fn from_bencode(bencode: &Bencode) -> Option<String> {
        match bencode {
            &ByteString(ref v) => std::str::from_utf8(v.as_slice()).map(|s| s.to_string()),
            _ => None
        }
    }
}

impl<T: ToBencode> ToBencode for Vec<T> {
    fn to_bencode(&self) -> Bencode { List(self.iter().map(|e| e.to_bencode()).collect()) }
}

impl<T: FromBencode> FromBencode for Vec<T> {
    fn from_bencode(bencode: &Bencode) -> Option<Vec<T>> {
        match bencode {
            &List(ref es) => {
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
        let mut m = TreeMap::new();
        for (key, value) in $m.iter() {
            m.insert(Key(Vec::from_slice(key.as_bytes())), value.to_bencode());
        }
        Dict(m)
    }}
}

macro_rules! map_from_bencode {
    ($mty:ident) => {{
        let res = match bencode {
            &Dict(ref map) => {
                let mut m = $mty::new();
                for (&Key(ref key), value) in map.iter() {
                    match str::from_utf8(key.as_slice()) {
                        Some(k) => {
                            let val: Option<T> = FromBencode::from_bencode(value);
                            match val {
                                Some(v) => m.insert(k.to_string(), v),
                                None => return None
                            }
                        }
                        None => return None
                    };
                }
                Some(m)
            }
            _ => None
        };
        res
    }}
}

impl<T: ToBencode> ToBencode for TreeMap<String, T> {
    fn to_bencode(&self) -> Bencode {
        map_to_bencode!(self)
    }
}

impl<T: FromBencode> FromBencode for TreeMap<String, T> {
    fn from_bencode(bencode: &Bencode) -> Option<TreeMap<String, T>> {
        map_from_bencode!(TreeMap)
    }
}

impl<T: ToBencode> ToBencode for HashMap<String, T> {
    fn to_bencode(&self) -> Bencode {
        map_to_bencode!(self)
    }
}

impl<T: FromBencode> FromBencode for HashMap<String, T> {
    fn from_bencode(bencode: &Bencode) -> Option<HashMap<String, T>> {
        map_from_bencode!(HashMap)
    }
}

pub fn from_buffer(buf: &[u8]) -> Result<Bencode, Error> {
    from_vec(Vec::from_slice(buf))
}

pub fn from_vec(buf: Vec<u8>) -> Result<Bencode, Error> {
    from_iter(buf.move_iter())
}

pub fn from_iter<T: Iterator<u8>>(iter: T) -> Result<Bencode, Error> {
    let streaming_parser = StreamingParser::new(iter);
    let mut parser = Parser::new(streaming_parser);
    parser.parse()
}

macro_rules! tryenc(($e:expr) => (
    match $e {
        Ok(e) => e,
        Err(e) => {
            self.error = Err(e);
            return
        }
    }
))

pub type EncoderResult<T> = IoResult<T>;

pub struct Encoder<'a> {
    writer: &'a mut io::Writer,
    writers: Vec<io::MemWriter>,
    expect_key: bool,
    keys: Vec<Key>,
    error: io::IoResult<()>,
    is_none: bool,
    stack: Vec<TreeMap<Key, Vec<u8>>>,
}

impl<'a> Encoder<'a> {
    pub fn new(writer: &'a mut io::Writer) -> Encoder<'a> {
        Encoder {
            writer: writer,
            writers: Vec::new(),
            expect_key: false,
            keys: Vec::new(),
            error: Ok(()),
            is_none: false,
            stack: Vec::new()
        }
    }

    pub fn buffer_encode<T: Encodable<Encoder<'a>, IoError>>(val: &T) -> EncoderResult<Vec<u8>> {
        let mut writer = io::MemWriter::new();
        {
            let mut encoder = Encoder::new(&mut writer);
            try!(val.encode(&mut encoder));
            if encoder.error.is_err() {
                return Err(encoder.error.unwrap_err())
            }
        }
        Ok(writer.unwrap())
    }

    fn get_writer(&'a mut self) -> &'a mut io::Writer {
        if self.writers.len() == 0 {
            &mut self.writer as &'a mut io::Writer
        } else {
            self.writers.mut_last().unwrap() as &'a mut io::Writer
        }
    }

    fn encode_dict(&mut self, dict: &TreeMap<Key, Vec<u8>>) -> EncoderResult<()> {
        try!(write!(self.get_writer(), "d"));
        for (key, value) in dict.iter() {
            try!(key.encode(self));
            try!(self.get_writer().write(value.as_slice()));
        }
        write!(self.get_writer(), "e")
    }

    fn error(&mut self, msg: &'static str) -> EncoderResult<()> {
        Err(IoError {
            kind: io::InvalidInput,
            desc: msg,
            detail: None
        })
    }
}

macro_rules! expect_value(() => {
    if self.expect_key {
        return self.error("Only 'string' map keys allowed");
    }
})

impl<'a> serialize::Encoder<IoError> for Encoder<'a> {
    fn emit_nil(&mut self) -> EncoderResult<()> { expect_value!(); write!(self.get_writer(), "0:") }

    fn emit_uint(&mut self, v: uint) -> EncoderResult<()> { self.emit_i64(v as i64) }

    fn emit_u8(&mut self, v: u8) -> EncoderResult<()> { self.emit_i64(v as i64) }

    fn emit_u16(&mut self, v: u16) -> EncoderResult<()> { self.emit_i64(v as i64) }

    fn emit_u32(&mut self, v: u32) -> EncoderResult<()> { self.emit_i64(v as i64) }

    fn emit_u64(&mut self, v: u64) -> EncoderResult<()> { self.emit_i64(v as i64) }

    fn emit_int(&mut self, v: int) -> EncoderResult<()> { self.emit_i64(v as i64) }

    fn emit_i8(&mut self, v: i8) -> EncoderResult<()> { self.emit_i64(v as i64) }

    fn emit_i16(&mut self, v: i16) -> EncoderResult<()> { self.emit_i64(v as i64) }

    fn emit_i32(&mut self, v: i32) -> EncoderResult<()> { self.emit_i64(v as i64) }

    fn emit_i64(&mut self, v: i64) -> EncoderResult<()> { expect_value!(); write!(self.get_writer(), "i{}e", v) }

    fn emit_bool(&mut self, v: bool) -> EncoderResult<()> {
        expect_value!();
        if v {
            self.emit_str("true")
        } else {
            self.emit_str("false")
        }
    }

    fn emit_f32(&mut self, v: f32) -> EncoderResult<()> {
        expect_value!();
        self.emit_str(std::f32::to_str_hex(v).as_slice())
    }

    fn emit_f64(&mut self, v: f64) -> EncoderResult<()> {
        expect_value!();
        self.emit_str(std::f64::to_str_hex(v).as_slice())
    }

    fn emit_char(&mut self, v: char) -> EncoderResult<()> {
        expect_value!();
        self.emit_str(str::from_char(v).as_slice())
    }

    fn emit_str(&mut self, v: &str) -> EncoderResult<()> {
        if self.expect_key {
            self.keys.push(Key(Vec::from_slice(v.as_bytes())));
            Ok(())
        } else {
            try!(write!(self.get_writer(), "{}:", v.len()));
            self.get_writer().write(v.as_bytes())
        }
    }

    fn emit_enum(&mut self, _name: &str, _f: |&mut Encoder<'a>| -> EncoderResult<()>) -> EncoderResult<()> {
        self.error("emit_enum not implemented")
    }

    fn emit_enum_variant(&mut self, _v_name: &str, _v_id: uint, _len: uint, _f: |&mut Encoder<'a>| -> EncoderResult<()>) -> EncoderResult<()> {
        self.error("emit_enum_variant not implemented")
    }

    fn emit_enum_variant_arg(&mut self, _a_idx: uint, _f: |&mut Encoder<'a>| -> EncoderResult<()>) -> EncoderResult<()> {
        self.error("emit_enum_variant_arg not implemented")
    }

    fn emit_enum_struct_variant(&mut self, _v_name: &str, _v_id: uint, _len: uint, _f: |&mut Encoder<'a>| -> EncoderResult<()>) -> EncoderResult<()> {
        self.error("emit_enum_struct_variant not implemented")
    }

    fn emit_enum_struct_variant_field(&mut self, _f_name: &str, _f_idx: uint, _f: |&mut Encoder<'a>| -> EncoderResult<()>) -> EncoderResult<()> {
        self.error("emit_enum_struct_variant_field not implemented")
    }

    fn emit_struct(&mut self, _name: &str, _len: uint, f: |&mut Encoder<'a>| -> EncoderResult<()>) -> EncoderResult<()> {
        expect_value!();
        self.stack.push(TreeMap::new());
        try!(f(self));
        let dict = self.stack.pop().unwrap();
        try!(self.encode_dict(&dict));
        self.is_none = false;
        Ok(())
    }

    fn emit_struct_field(&mut self, f_name: &str, _f_idx: uint, f: |&mut Encoder<'a>| -> EncoderResult<()>) -> EncoderResult<()> {
        expect_value!();
        self.writers.push(io::MemWriter::new());
        try!(f(self));
        let data = self.writers.pop().unwrap();
        let dict = self.stack.mut_last().unwrap();
        if !self.is_none {
            dict.insert(Key(Vec::from_slice(f_name.as_bytes())), data.unwrap());
        }
        self.is_none = false;
        Ok(())
    }

    fn emit_tuple(&mut self, _len: uint, _f: |&mut Encoder<'a>| -> EncoderResult<()>) -> EncoderResult<()> {
        self.error("emit_tuple not implemented")
    }

    fn emit_tuple_arg(&mut self, _idx: uint, _f: |&mut Encoder<'a>| -> EncoderResult<()>) -> EncoderResult<()> {
        self.error("emit_tuple_arg not implemented")
    }
    fn emit_tuple_struct(&mut self, _name: &str, _len: uint, _f: |&mut Encoder<'a>| -> EncoderResult<()>) -> EncoderResult<()> {
        self.error("emit_tuple_struct not implemented")
    }
    fn emit_tuple_struct_arg(&mut self, _f_idx: uint, _f: |&mut Encoder<'a>| -> EncoderResult<()>) -> EncoderResult<()> {
        self.error("emit_tuple_struct_arg not implemented")
    }

    fn emit_option(&mut self, f: |&mut Encoder<'a>| -> EncoderResult<()>) -> EncoderResult<()> {
        expect_value!();
        f(self)
    }

    fn emit_option_none(&mut self) -> EncoderResult<()> {
        expect_value!();
        self.is_none = true;
        write!(self.get_writer(), "3:nil")
    }

    fn emit_option_some(&mut self, f: |&mut Encoder<'a>| -> EncoderResult<()>) -> EncoderResult<()> {
        expect_value!();
        f(self)
    }

    fn emit_seq(&mut self, _len: uint, f: |this: &mut Encoder<'a>| -> EncoderResult<()>) -> EncoderResult<()> {
        expect_value!();
        try!(write!(self.get_writer(), "l"));
        try!(f(self));
        self.is_none = false;
        write!(self.get_writer(), "e")
    }

    fn emit_seq_elt(&mut self, _idx: uint, f: |this: &mut Encoder<'a>| -> EncoderResult<()>) -> EncoderResult<()> {
        expect_value!();
        try!(f(self));
        self.is_none = false;
        Ok(())
    }

    fn emit_map(&mut self, _len: uint, f: |&mut Encoder<'a>| -> EncoderResult<()>) -> EncoderResult<()> {
        expect_value!();
        self.stack.push(TreeMap::new());
        try!(f(self));
        let dict = self.stack.pop().unwrap();
        try!(self.encode_dict(&dict));
        self.is_none = false;
        Ok(())
    }

    fn emit_map_elt_key(&mut self, _idx: uint, f: |&mut Encoder<'a>| -> EncoderResult<()>) -> EncoderResult<()> {
        expect_value!();
        self.writers.push(io::MemWriter::new());
        self.expect_key = true;
        try!(f(self));
        self.expect_key = false;
        self.is_none = false;
        Ok(())
    }

    fn emit_map_elt_val(&mut self, _idx: uint, f: |&mut Encoder<'a>| -> EncoderResult<()>) -> EncoderResult<()> {
        expect_value!();
        try!(f(self));
        let key = self.keys.pop();
        let data = self.writers.pop().unwrap();
        let dict = self.stack.mut_last().unwrap();
        dict.insert(key.unwrap(), data.unwrap());
        self.is_none = false;
        Ok(())
    }
}

pub struct Parser<T> {
    reader: T,
    depth: u32,
}

impl<T: Iterator<BencodeEvent>> Parser<T> {
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
            Some(NumberValue(v)) => Ok(Number(v)),
            Some(ByteStringValue(v)) => Ok(ByteString(v)),
            Some(ListStart) => self.parse_list(current),
            Some(DictStart) => self.parse_dict(current),
            Some(ParseError(err)) => Err(err),
            None => Ok(Empty),
            x => fail!("[root] Unreachable but got {}", x)
        };
        if self.depth == 0 {
            let next = self.reader.next();
            match res {
                Err(_) => res,
                _ => {
                    match next {
                        Some(ParseError(err)) => Err(err),
                        None => res,
                        x => fail!("Unreachable but got {}", x)
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
                x => fail!("[list] Unreachable but got {}", x)
            }
        }
        self.depth -= 1;
        Ok(List(list))
    }

    fn parse_dict(&mut self, mut current: Option<BencodeEvent>) -> Result<Bencode, Error> {
        self.depth += 1;
        let mut map = TreeMap::new();
        loop {
            current = self.reader.next();
            let key = match current {
                Some(DictEnd) => break,
                Some(DictKey(v)) => Key(v),
                Some(ParseError(err)) => return Err(err),
                x => fail!("[dict] Unreachable but got {}", x)
            };
            current = self.reader.next();
            let value = try!(self.parse_elem(current));
            map.insert(key, value);
        }
        self.depth -= 1;
        Ok(Dict(map))
    }
}

macro_rules! dec_expect_value(() => {
    if self.expect_key {
        return Err(Message("Only 'string' map keys allowed".to_string()))
    }
})

static EMPTY: Bencode = Empty;

#[deriving(Eq, PartialEq, Clone, Show)]
pub enum DecoderError {
    Message(String),
    Expecting(&'static str, String),
    Unimplemented(&'static str),
}

pub type DecoderResult<T> = Result<T, DecoderError>;

pub struct Decoder<'a> {
    keys: Vec<Key>,
    expect_key: bool,
    stack: Vec<&'a Bencode>,
}

impl<'a> Decoder<'a> {
    pub fn new(bencode: &'a Bencode) -> Decoder<'a> {
        Decoder {
            keys: Vec::new(),
            expect_key: false,
            stack: Vec::from_slice([bencode])
        }
    }

    fn try_read<T: FromBencode>(&mut self, ty: &'static str) -> DecoderResult<T> {
        let val = self.stack.pop();
        match val.and_then(|b| FromBencode::from_bencode(b)) {
            Some(v) => Ok(v),
            None => self.error(format!("Error decoding value as '{}': {}", ty, val))
        }
    }

    fn error<T>(&self, msg: String) -> DecoderResult<T> {
        Err(Message(msg))
    }

    fn unimplemented<T>(&self, m: &'static str) -> DecoderResult<T> {
        Err(Unimplemented(m))
    }
}

impl<'a> serialize::Decoder<DecoderError> for Decoder<'a> {
    fn read_nil(&mut self) -> DecoderResult<()> {
        dec_expect_value!();
        self.try_read("nil")
    }

    fn read_uint(&mut self) -> DecoderResult<uint> {
        dec_expect_value!();
        self.try_read("uint")
    }

    fn read_u8(&mut self) -> DecoderResult<u8> {
        dec_expect_value!();
        self.try_read("u8")
    }

    fn read_u16(&mut self) -> DecoderResult<u16> {
        dec_expect_value!();
        self.try_read("u16")
    }

    fn read_u32(&mut self) -> DecoderResult<u32> {
        dec_expect_value!();
        self.try_read("u32")
    }

    fn read_u64(&mut self) -> DecoderResult<u64> {
        dec_expect_value!();
        self.try_read("u64")
    }

    fn read_int(&mut self) -> DecoderResult<int> {
        dec_expect_value!();
        self.try_read("int")
    }

    fn read_i8(&mut self) -> DecoderResult<i8> {
        dec_expect_value!();
        self.try_read("i8")
    }

    fn read_i16(&mut self) -> DecoderResult<i16> {
        dec_expect_value!();
        self.try_read("i16")
    }

    fn read_i32(&mut self) -> DecoderResult<i32> {
        dec_expect_value!();
        self.try_read("i32")
    }

    fn read_i64(&mut self) -> DecoderResult<i64> {
        dec_expect_value!();
        self.try_read("i64")
    }

    fn read_bool(&mut self) -> DecoderResult<bool> {
        dec_expect_value!();
        self.try_read("bool")
    }

    fn read_f32(&mut self) -> DecoderResult<f32> {
        dec_expect_value!();
        self.try_read("f32")
    }

    fn read_f64(&mut self) -> DecoderResult<f64> {
        dec_expect_value!();
        self.try_read("f64")
    }

    fn read_char(&mut self) -> DecoderResult<char> {
        dec_expect_value!();
        self.try_read("char")
    }

    fn read_str(&mut self) -> DecoderResult<String> {
        if self.expect_key {
            let Key(b) = self.keys.pop().unwrap();
            match str::from_utf8_owned(Vec::from_slice(b.as_slice())) {
                Ok(s) => Ok(s),
                Err(_) => self.error("error decoding key as utf-8".to_string())
            }
        } else {
            self.try_read("str")
        }
    }

    fn read_enum<T>(&mut self, _name: &str, _f: |&mut Decoder<'a>| -> DecoderResult<T>) -> DecoderResult<T> {
        self.unimplemented("read_enum")
    }

    fn read_enum_variant<T>(&mut self, _names: &[&str], _f: |&mut Decoder<'a>, uint| -> DecoderResult<T>) -> DecoderResult<T> {
        self.unimplemented("read_enum_variant")
    }

    fn read_enum_variant_arg<T>(&mut self, _a_idx: uint, _f: |&mut Decoder<'a>| -> DecoderResult<T>) -> DecoderResult<T> {
        self.unimplemented("read_enum_variant_arg")
    }

    fn read_enum_struct_variant<T>(&mut self, _names: &[&str], _f: |&mut Decoder<'a>, uint| -> DecoderResult<T>) -> DecoderResult<T> {
        self.unimplemented("read_enum_struct_variant")
    }

    fn read_enum_struct_variant_field<T>(&mut self, _f_name: &str, _f_idx: uint, _f: |&mut Decoder<'a>| -> DecoderResult<T>) -> DecoderResult<T> {
        self.unimplemented("read_enum_struct_variant_field")
    }

    fn read_struct<T>(&mut self, _s_name: &str, _len: uint, f: |&mut Decoder<'a>| -> DecoderResult<T>) -> DecoderResult<T> {
        dec_expect_value!();
        let res = try!(f(self));
        self.stack.pop();
        Ok(res)
    }

    fn read_struct_field<T>(&mut self, f_name: &str, _f_idx: uint, f: |&mut Decoder<'a>| -> DecoderResult<T>) -> DecoderResult<T> {
        dec_expect_value!();
        let val = match self.stack.last() {
            Some(v) => {
                match *v {
                    &Dict(ref m) => {
                        match m.find(&Key(Vec::from_slice(f_name.as_bytes()))) {
                            Some(v) => v,
                            None => &EMPTY
                        }
                    }
                    _ => return Err(Expecting("Dict", v.to_str()))
                }
            }
            None => return Err(Expecting("Dict", "None".to_string()))
        };
        self.stack.push(val);
        f(self)
    }

    fn read_tuple<T>(&mut self, _f: |&mut Decoder<'a>, uint| -> DecoderResult<T>) -> DecoderResult<T> {
        self.unimplemented("read_tuple")
    }

    fn read_tuple_arg<T>(&mut self, _a_idx: uint, _f: |&mut Decoder<'a>| -> DecoderResult<T>) -> DecoderResult<T> {
        self.unimplemented("read_tuple_arg")
    }

    fn read_tuple_struct<T>(&mut self, _s_name: &str, _f: |&mut Decoder<'a>, uint| -> DecoderResult<T>) -> DecoderResult<T> {
        self.unimplemented("read_tuple_struct")
    }

    fn read_tuple_struct_arg<T>(&mut self, _a_idx: uint, _f: |&mut Decoder<'a>| -> DecoderResult<T>) -> DecoderResult<T> {
        self.unimplemented("read_tuple_struct_arg")
    }

    fn read_option<T>(&mut self, f: |&mut Decoder<'a>, bool| -> DecoderResult<T>) -> DecoderResult<T> {
        match self.stack.pop() {
            Some(&Empty) => f(self, false),
            Some(b@&ByteString(ref v)) => {
                if v.as_slice() == bytes!("nil") {
                    f(self, false)
                } else {
                    self.stack.push(b);
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

    fn read_seq<T>(&mut self, f: |&mut Decoder<'a>, uint| -> DecoderResult<T>) -> DecoderResult<T> {
        dec_expect_value!();
        let len = match self.stack.pop() {
            Some(&List(ref list)) => {
                for v in list.as_slice().iter().rev() {
                    self.stack.push(v);
                }
                list.len()
            }
            val => return Err(Expecting("List", val.to_str()))
        };
        f(self, len)
    }

    fn read_seq_elt<T>(&mut self, _idx: uint, f: |&mut Decoder<'a>| -> DecoderResult<T>) -> DecoderResult<T> {
        dec_expect_value!();
        f(self)
    }

    fn read_map<T>(&mut self, f: |&mut Decoder<'a>, uint| -> DecoderResult<T>) -> DecoderResult<T> {
        dec_expect_value!();
        let len = match self.stack.pop() {
            Some(&Dict(ref m)) => {
                for (key, value) in m.iter() {
                    self.keys.push(key.clone());
                    self.stack.push(value);
                }
                m.len()
            }
            val => return Err(Expecting("Dict", val.to_str()))
        };
        f(self, len)
    }

    fn read_map_elt_key<T>(&mut self, _idx: uint, f: |&mut Decoder<'a>| -> DecoderResult<T>) -> DecoderResult<T> {
        dec_expect_value!();
        self.expect_key = true;
        let res = try!(f(self));
        self.expect_key = false;
        Ok(res)
    }

    fn read_map_elt_val<T>(&mut self, _idx: uint, f: |&mut Decoder<'a>| -> DecoderResult<T>) -> DecoderResult<T> {
        dec_expect_value!();
        f(self)
    }
}

#[cfg(test)]
mod tests {
    use serialize::{Encodable, Decodable};
    use std::collections::treemap::TreeMap;
    use std::collections::hashmap::HashMap;

    use streaming::Error;
    use streaming::{BencodeEvent, NumberValue, ByteStringValue, ListStart,
                    ListEnd, DictStart, DictKey, DictEnd, ParseError};

    use super::{Bencode, ToBencode};
    use super::{Encoder, ByteString, List, Number, Dict, Key, Empty};
    use super::{Parser, Decoder, DecoderResult};

    macro_rules! assert_encoding(($value:expr, $expected:expr) => ({
        let value = $value;
        let encoded = match Encoder::buffer_encode(&value) {
            Ok(e) => e,
            Err(err) => fail!("Unexpected failure: {}", err)
        };
        assert_eq!($expected.as_slice(), encoded.as_slice());
    }))

    macro_rules! assert_decoding(($enc:expr, $value:expr) => ({
        let bencode = super::from_vec($enc).unwrap();
        let mut decoder = Decoder::new(&bencode);
        let result = Decodable::decode(&mut decoder);
        assert_eq!(Ok($value), result);
    }))

    macro_rules! gen_encode_test(($name:ident, $($val:expr -> $enc:expr),+) => {
        #[test]
        fn $name() {
            $(assert_encoding!($val, $enc);)+
        }
    })

    macro_rules! gen_tobencode_test(($name:ident, $($val:expr -> $enc:expr),+) => {
        #[test]
        fn $name() {
            $({
                let value = $val.to_bencode();
                assert_encoding!(value, $enc)
            };)+
        }
    })

    macro_rules! assert_identity(($value:expr) => ({
        let value = $value;
        let encoded = match Encoder::buffer_encode(&value) {
            Ok(e) => e,
            Err(err) => fail!("Unexpected failure: {}", err)
        };
        let bencode = super::from_vec(encoded).unwrap();
        let mut decoder = Decoder::new(&bencode);
        let result = Decodable::decode(&mut decoder);
        assert_eq!(Ok(value), result);
    }))

    macro_rules! gen_identity_test(($name:ident, $($val:expr),+) => {
        #[test]
        fn $name() {
            $(assert_identity!($val);)+
        }
    })

    macro_rules! gen_encode_identity_test(($name_enc:ident, $name_ident:ident, $($val:expr -> $enc:expr),+) => {
        gen_encode_test!($name_enc, $($val -> $enc),+)
        gen_identity_test!($name_ident, $($val),+)
    })

    macro_rules! gen_complete_test(($name_enc:ident, $name_benc:ident, $name_ident:ident, $($val:expr -> $enc:expr),+) => {
        gen_encode_test!($name_enc, $($val -> $enc),+)
        gen_tobencode_test!($name_benc, $($val -> $enc),+)
        gen_identity_test!($name_ident, $($val),+)
    })

    fn bytes(s: &str) -> Vec<u8> {
        Vec::from_slice(s.as_bytes())
    }

    gen_complete_test!(encodes_unit,
                       tobencode_unit,
                       identity_unit,
                       () -> bytes("0:"))

    gen_complete_test!(encodes_option_none,
                       tobencode_option_none,
                       identity_option_none,
                       {
                           let none: Option<int> = None;
                           none
                       } -> bytes("3:nil"))

    gen_complete_test!(encodes_option_some,
                       tobencode_option_some,
                       identity_option_some,
                       Some(1) -> bytes("i1e"),
                       Some("rust".to_string()) -> bytes("4:rust"),
                       Some(vec![(), ()]) -> bytes("l0:0:e"))

    gen_complete_test!(encodes_nested_option,
                       tobencode_nested_option,
                       identity_nested_option,
                       Some(Some(1)) -> bytes("i1e"),
                       Some(Some("rust".to_string())) -> bytes("4:rust"))

    #[test]
    fn option_is_none_if_any_nested_option_is_none() {
        let value: Option<Option<int>> = Some(None);
        let encoded = match Encoder::buffer_encode(&value) {
            Ok(e) => e,
            Err(err) => fail!("Unexpected failure: {}", err)
        };
        let none: Option<Option<int>> = None;
        assert_decoding!(encoded, none);
    }

    gen_complete_test!(encodes_zero_int,
                       tobencode_zero_int,
                       identity_zero_int,
                       0i -> bytes("i0e"))

    gen_complete_test!(encodes_positive_int,
                       tobencode_positive_int,
                       identity_positive_int,
                       5i -> bytes("i5e"),
                       99i -> bytes("i99e"),
                       ::std::int::MAX -> bytes(format!("i{}e", ::std::int::MAX).as_slice()))

    gen_complete_test!(encodes_negative_int,
                       tobencode_negative_int,
                       identity_negative_int,
                       -5i -> bytes("i-5e"),
                       -99i -> bytes("i-99e"),
                       ::std::int::MIN -> bytes(format!("i{}e", ::std::int::MIN).as_slice()))

    gen_complete_test!(encodes_zero_i8,
                       tobencode_zero_i8,
                       identity_zero_i8,
                       0i8 -> bytes("i0e"))

    gen_complete_test!(encodes_positive_i8,
                       tobencode_positive_i8,
                       identity_positive_i8,
                       5i8 -> bytes("i5e"),
                       99i8 -> bytes("i99e"),
                       ::std::i8::MAX -> bytes(format!("i{}e", ::std::i8::MAX).as_slice()))

    gen_complete_test!(encodes_negative_i8,
                       tobencode_negative_i8,
                       identity_negative_i8,
                       -5i8 -> bytes("i-5e"),
                       -99i8 -> bytes("i-99e"),
                       ::std::i8::MIN -> bytes(format!("i{}e", ::std::i8::MIN).as_slice()))

    gen_complete_test!(encodes_zero_i16,
                       tobencode_zero_i16,
                       identity_zero_i16,
                       0i16 -> bytes("i0e"))

    gen_complete_test!(encodes_positive_i16,
                       tobencode_positive_i16,
                       identity_positive_i16,
                       5i16 -> bytes("i5e"),
                       99i16 -> bytes("i99e"),
                       ::std::i16::MAX -> bytes(format!("i{}e", ::std::i16::MAX).as_slice()))

    gen_complete_test!(encodes_negative_i16,
                       tobencode_negative_i16,
                       identity_negative_i16,
                       -5i16 -> bytes("i-5e"),
                       -99i16 -> bytes("i-99e"),
                       ::std::i16::MIN -> bytes(format!("i{}e", ::std::i16::MIN).as_slice()))

    gen_complete_test!(encodes_zero_i32,
                       tobencode_zero_i32,
                       identity_zero_i32,
                       0i32 -> bytes("i0e"))

    gen_complete_test!(encodes_positive_i32,
                       tobencode_positive_i32,
                       identity_positive_i32,
                       5i32 -> bytes("i5e"),
                       99i32 -> bytes("i99e"),
                       ::std::i32::MAX -> bytes(format!("i{}e", ::std::i32::MAX).as_slice()))

    gen_complete_test!(encodes_negative_i32,
                       tobencode_negative_i32,
                       identity_negative_i32,
                       -5i32 -> bytes("i-5e"),
                       -99i32 -> bytes("i-99e"),
                       ::std::i32::MIN -> bytes(format!("i{}e", ::std::i32::MIN).as_slice()))

    gen_complete_test!(encodes_zero_i64,
                       tobencode_zero_i64,
                       identity_zero_i64,
                       0i64 -> bytes("i0e"))

    gen_complete_test!(encodes_positive_i64,
                       tobencode_positive_i64,
                       identity_positive_i64,
                       5i64 -> bytes("i5e"),
                       99i64 -> bytes("i99e"),
                       ::std::i64::MAX -> bytes(format!("i{}e", ::std::i64::MAX).as_slice()))

    gen_complete_test!(encodes_negative_i64,
                       tobencode_negative_i64,
                       identity_negative_i64,
                       -5i64 -> bytes("i-5e"),
                       -99i64 -> bytes("i-99e"),
                       ::std::i64::MIN -> bytes(format!("i{}e", ::std::i64::MIN).as_slice()))

    gen_complete_test!(encodes_zero_uint,
                       tobencode_zero_uint,
                       identity_zero_uint,
                       0u -> bytes("i0e"))

    gen_complete_test!(encodes_positive_uint,
                       tobencode_positive_uint,
                       identity_positive_uint,
                       5u -> bytes("i5e"),
                       99u -> bytes("i99e"),
                       ::std::uint::MAX / 2 -> bytes(format!("i{}e", ::std::uint::MAX / 2).as_slice()))

    gen_complete_test!(encodes_zero_u8,
                       tobencode_zero_u8,
                       identity_zero_u8,
                       0u8 -> bytes("i0e"))

    gen_complete_test!(encodes_positive_u8,
                       tobencode_positive_u8,
                       identity_positive_u8,
                       5u8 -> bytes("i5e"),
                       99u8 -> bytes("i99e"),
                       ::std::u8::MAX -> bytes(format!("i{}e", ::std::u8::MAX).as_slice()))

    gen_complete_test!(encodes_zero_u16,
                       tobencode_zero_u16,
                       identity_zero_u16,
                       0u16 -> bytes("i0e"))

    gen_complete_test!(encodes_positive_u16,
                       tobencode_positive_u16,
                       identity_positive_u16,
                       5u16 -> bytes("i5e"),
                       99u16 -> bytes("i99e"),
                       ::std::u16::MAX -> bytes(format!("i{}e", ::std::u16::MAX).as_slice()))

    gen_complete_test!(encodes_zero_u32,
                       tobencode_zero_u32,
                       identity_zero_u32,
                       0u32 -> bytes("i0e"))

    gen_complete_test!(encodes_positive_u32,
                       tobencode_positive_u32,
                       identity_positive_u32,
                       5u32 -> bytes("i5e"),
                       99u32 -> bytes("i99e"),
                       ::std::u32::MAX -> bytes(format!("i{}e", ::std::u32::MAX).as_slice()))

    gen_complete_test!(encodes_zero_u64,
                       tobencode_zero_u64,
                       identity_zero_u64,
                       0u64 -> bytes("i0e"))

    gen_complete_test!(encodes_positive_u64,
                       tobencode_positive_u64,
                       identity_positive_u64,
                       5u64 -> bytes("i5e"),
                       99u64 -> bytes("i99e"),
                       ::std::u64::MAX / 2 -> bytes(format!("i{}e", ::std::u64::MAX / 2).as_slice()))

    gen_complete_test!(encodes_bool,
                       tobencode_bool,
                       identity_bool,
                       true -> bytes("4:true"),
                       false -> bytes("5:false"))

    gen_complete_test!(encodes_zero_f32,
                       tobencode_zero_f32,
                       identity_zero_f32,
                       0.0f32 -> bytes("1:0"))

    gen_complete_test!(encodes_positive_f32,
                       tobencode_positive_f32,
                       identity_positive_f32,
                       99.0f32 -> bytes("2:63"),
                       101.12345f32 -> bytes("8:65.1f9a8"))

    gen_complete_test!(encodes_negative_f32,
                       tobencode_negative_f32,
                       identity_negative_f32,
                       -99.0f32 -> bytes("3:-63"),
                       -101.12345f32 -> bytes("9:-65.1f9a8"))

    gen_complete_test!(encodes_zero_f64,
                       tobencode_zero_f64,
                       identity_zero_f64,
                       0.0f64 -> bytes("1:0"))

    gen_complete_test!(encodes_positive_f64,
                       tobencode_positive_f64,
                       identity_positive_f64,
                       99.0f64 -> bytes("2:63"),
                       101.12345f64 -> bytes("15:65.1f9a6b50b0f4"))

    gen_complete_test!(encodes_negative_f64,
                       tobencode_negative_f64,
                       identity_negative_f64,
                       -99.0f64 -> bytes("3:-63"),
                       -101.12345f64 -> bytes("16:-65.1f9a6b50b0f4"))

    gen_complete_test!(encodes_lower_letter_char,
                       tobencode_lower_letter_char,
                       identity_lower_letter_char,
                       'a' -> bytes("1:a"),
                       'c' -> bytes("1:c"),
                       'z' -> bytes("1:z"))

    gen_complete_test!(encodes_upper_letter_char,
                       tobencode_upper_letter_char,
                       identity_upper_letter_char,
                       'A' -> bytes("1:A"),
                       'C' -> bytes("1:C"),
                       'Z' -> bytes("1:Z"))

    gen_complete_test!(encodes_multibyte_char,
                       tobencode_multibyte_char,
                       identity_multibyte_char,
                       'ệ' -> bytes("3:ệ"),
                       '虎' -> bytes("3:虎"))

    gen_complete_test!(encodes_control_char,
                       tobencode_control_char,
                       identity_control_char,
                       '\n' -> bytes("1:\n"),
                       '\r' -> bytes("1:\r"),
                       '\0' -> bytes("1:\0"))

    gen_complete_test!(encode_empty_str,
                      tobencode_empty_str,
                      identity_empty_str,
                      "".to_string() -> bytes("0:"))

    gen_complete_test!(encode_str,
                      tobencode_str,
                      identity_str,
                      "a".to_string() -> bytes("1:a"),
                      "foo".to_string() -> bytes("3:foo"),
                      "This is nice!?#$%".to_string() -> bytes("17:This is nice!?#$%"))

    gen_complete_test!(encode_str_with_multibyte_chars,
                      tobencode_str_with_multibyte_chars,
                      identity_str_with_multibyte_chars,
                      "Löwe 老虎 Léopard".to_string() -> bytes("21:Löwe 老虎 Léopard"),
                      "いろはにほへとちりぬるを".to_string() -> bytes("36:いろはにほへとちりぬるを"))

    gen_complete_test!(encodes_empty_vec,
                       tobencode_empty_vec,
                       identity_empty_vec,
                       {
                           let empty: Vec<u8> = Vec::new();
                           empty
                       } -> bytes("le"))

    gen_complete_test!(encodes_nonmpty_vec,
                       tobencode_nonmpty_vec,
                       identity_nonmpty_vec,
                       vec![0, 1, 3, 4] -> bytes("li0ei1ei3ei4ee"),
                       vec!["foo".to_string(), "b".to_string()] -> bytes("l3:foo1:be"))

    gen_complete_test!(encodes_nested_vec,
                       tobencode_nested_vec,
                       identity_nested_vec,
                       vec![vec![1], vec![2, 3], vec![]] -> bytes("lli1eeli2ei3eelee"))

    #[deriving(Eq, PartialEq, Show, Encodable, Decodable)]
    struct SimpleStruct {
        a: uint,
        b: ~[String],
    }

    #[deriving(Eq, PartialEq, Show, Encodable, Decodable)]
    struct InnerStruct {
        field_one: (),
        list: ~[uint],
        abc: String
    }

    #[deriving(Eq, PartialEq, Show, Encodable, Decodable)]
    struct OuterStruct {
        inner: ~[InnerStruct],
        is_true: bool
    }

    gen_encode_identity_test!(encodes_struct,
                              identity_struct,
                              SimpleStruct {
                                  b: ~["foo".to_string(), "baar".to_string()],
                                  a: 123
                              } -> bytes("d1:ai123e1:bl3:foo4:baaree"),
                              SimpleStruct {
                                  a: 1234567890,
                                  b: ~[]
                              } -> bytes("d1:ai1234567890e1:blee"))

    gen_encode_identity_test!(encodes_nested_struct,
                              identity_nested_struct,
                              OuterStruct {
                                  is_true: true,
                                  inner: ~[InnerStruct {
                                      field_one: (),
                                      list: ~[99, 5],
                                      abc: "rust".to_string()
                                  }, InnerStruct {
                                      field_one: (),
                                      list: ~[],
                                      abc: "".to_string()
                                  }]
                              } -> bytes("d\
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
                                          e"))

    macro_rules! map(($m:ident, $(($key:expr, $val:expr)),*) => {{
        let mut _m = $m::new();
        $(_m.insert($key, $val);)*
        _m
    }})

    gen_complete_test!(encodes_hashmap,
                       bencode_hashmap,
                       identity_hashmap,
                       map!(HashMap, ("a".to_string(), 1)) -> bytes("d1:ai1ee"),
                       map!(HashMap, ("foo".to_string(), "a".to_string()), ("bar".to_string(), "bb".to_string())) -> bytes("d3:bar2:bb3:foo1:ae"))

    gen_complete_test!(encodes_nested_hashmap,
                       bencode_nested_hashmap,
                       identity_nested_hashmap,
                       map!(HashMap, ("a".to_string(), map!(HashMap, ("foo".to_string(), 101), ("bar".to_string(), 102)))) -> bytes("d1:ad3:bari102e3:fooi101eee"))
    #[test]
    fn decode_error_on_wrong_map_key_type() {
        let benc = Dict(map!(TreeMap, (Key(bytes("foo")), ByteString(bytes("bar")))));
        let mut decoder = Decoder::new(&benc);
        let res: DecoderResult<TreeMap<int, String>> = Decodable::decode(&mut decoder);
        assert!(res.is_err());
    }

    #[test]
    fn encode_error_on_wrong_map_key_type() {
        let m = map!(HashMap, (1, "foo"));
        let encoded = Encoder::buffer_encode(&m);
        assert!(encoded.is_err())
    }

    #[test]
    fn encodes_struct_fields_in_sorted_order() {
        #[deriving(Encodable)]
        struct OrderedStruct {
            z: int,
            a: int,
            ab: int,
            aa: int,
        }
        let s = OrderedStruct {
            z: 4,
            a: 1,
            ab: 3,
            aa: 2
        };
        assert_eq!(Encoder::buffer_encode(&s), Ok(bytes("d1:ai1e2:aai2e2:abi3e1:zi4ee")));
    }

    #[deriving(Encodable, Decodable, Eq, PartialEq, Show, Clone)]
    struct OptionalStruct {
        a: Option<int>,
        b: int,
        c: Option<~[Option<bool>]>,
    }

    #[deriving(Encodable, Decodable, Eq, PartialEq, Show)]
    struct OptionalStructOuter {
        a: Option<OptionalStruct>,
        b: Option<int>,
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
                              {
                                  OptionalStructOuter {
                                      a: Some(OPT_STRUCT.clone()),
                                      b: None
                                  }
                              } -> bytes("d1:ad1:bi10eee"),
                              {
                                  let a = OptionalStruct {
                                      a: None,
                                      b: 10,
                                      c: Some(~[Some(true), None])
                                  };
                                  OptionalStructOuter {
                                      a: Some(a),
                                      b: Some(99)
                                  }
                              } -> bytes("d1:ad1:bi10e1:cl4:true3:nilee1:bi99ee"))

    fn try_bencode(bencode: Bencode) -> Vec<u8> {
        match bencode.to_bytes() {
            Ok(v) => v,
            Err(err) => fail!("Unexpected error: {}", err)
        }
    }

    #[test]
    fn encodes_empty_bytestring() {
        assert_eq!(try_bencode(ByteString(Vec::new())), bytes("0:"));
    }

    #[test]
    fn encodes_nonempty_bytestring() {
        assert_eq!(try_bencode(ByteString(Vec::from_slice(bytes!("abc")))), bytes("3:abc"));
        assert_eq!(try_bencode(ByteString(vec![0, 1, 2, 3])), bytes("4:").append([0u8, 1, 2, 3]));
    }

    #[test]
    fn encodes_empty_list() {
        assert_eq!(try_bencode(List(Vec::new())), bytes("le"));
    }

    #[test]
    fn encodes_nonempty_list() {
        assert_eq!(try_bencode(List(vec![Number(1)])), bytes("li1ee"));
        assert_eq!(try_bencode(List(vec![ByteString(Vec::from_slice("foobar".as_bytes())),
                          Number(-1)])), bytes("l6:foobari-1ee"));
    }

    #[test]
    fn encodes_nested_list() {
        assert_eq!(try_bencode(List(vec![List(vec![])])), bytes("llee"));
        let list = List(vec![Number(1988), List(vec![Number(2014)])]);
        assert_eq!(try_bencode(list), bytes("li1988eli2014eee"));
    }

    #[test]
    fn encodes_empty_dict() {
        assert_eq!(try_bencode(Dict(TreeMap::new())), bytes("de"));
    }

    #[test]
    fn encodes_dict_with_items() {
        let mut m = TreeMap::new();
        m.insert(Key::from_str("k1"), Number(1));
        assert_eq!(try_bencode(Dict(m.clone())), bytes("d2:k1i1ee"));
        m.insert(Key::from_str("k2"), ByteString(vec![0, 0]));
        assert_eq!(try_bencode(Dict(m.clone())), bytes("d2:k1i1e2:k22:\0\0e"));
    }

    #[test]
    fn encodes_nested_dict() {
        let mut outer = TreeMap::new();
        let mut inner = TreeMap::new();
        inner.insert(Key::from_str("val"), ByteString(vec![68, 0, 90]));
        outer.insert(Key::from_str("inner"), Dict(inner));
        assert_eq!(try_bencode(Dict(outer)), bytes("d5:innerd3:val3:D\0Zee"));
    }

    #[test]
    fn encodes_dict_fields_in_sorted_order() {
        let mut m = TreeMap::new();
        m.insert(Key::from_str("z"), Number(1));
        m.insert(Key::from_str("abd"), Number(3));
        m.insert(Key::from_str("abc"), Number(2));
        assert_eq!(try_bencode(Dict(m)), bytes("d3:abci2e3:abdi3e1:zi1ee"));
    }

    fn assert_decoded_eq(events: &[BencodeEvent], expected: Result<Bencode, Error>) {
        let mut parser = Parser::new(events.to_owned().move_iter());
        let result = parser.parse();
        assert_eq!(expected, result);
    }

    #[test]
    fn decodes_empty_input() {
        assert_decoded_eq([], Ok(Empty));
    }

    #[test]
    fn decodes_number() {
        assert_decoded_eq([NumberValue(25)], Ok(Number(25)));
    }

    #[test]
    fn decodes_bytestring() {
        assert_decoded_eq([ByteStringValue(bytes("foo"))], Ok(ByteString(bytes("foo"))));
    }

    #[test]
    fn decodes_empty_list() {
        assert_decoded_eq([ListStart, ListEnd], Ok(List(vec![])));
    }

    #[test]
    fn decodes_list_with_elements() {
        assert_decoded_eq([ListStart,
                           NumberValue(1),
                           ListEnd], Ok(List(vec![Number(1)])));
        assert_decoded_eq([ListStart,
                           ByteStringValue(bytes("str")),
                           NumberValue(11),
                           ListEnd], Ok(List(vec![ByteString(bytes("str")),
                                               Number(11)])));
    }

    #[test]
    fn decodes_nested_list() {
        assert_decoded_eq([ListStart,
                           ListStart,
                           NumberValue(13),
                           ListEnd,
                           ByteStringValue(bytes("rust")),
                           ListEnd],
                           Ok(List(vec![List(vec![Number(13)]),
                                     ByteString(bytes("rust"))])));
    }

    #[test]
    fn decodes_empty_dict() {
        assert_decoded_eq([DictStart, DictEnd], Ok(Dict(TreeMap::new())));
    }

    #[test]
    fn decodes_dict_with_value() {
        let mut map = TreeMap::new();
        map.insert(Key::from_str("foo"), ByteString(bytes("rust")));
        assert_decoded_eq([DictStart,
                           DictKey(bytes("foo")),
                           ByteStringValue(bytes("rust")),
                           DictEnd], Ok(Dict(map)));
    }

    #[test]
    fn decodes_dict_with_values() {
        let mut map = TreeMap::new();
        map.insert(Key::from_str("num"), Number(9));
        map.insert(Key::from_str("str"), ByteString(bytes("abc")));
        map.insert(Key::from_str("list"), List(vec![Number(99)]));
        assert_decoded_eq([DictStart,
                           DictKey(bytes("num")),
                           NumberValue(9),
                           DictKey(bytes("str")),
                           ByteStringValue(bytes("abc")),
                           DictKey(bytes("list")),
                           ListStart,
                           NumberValue(99),
                           ListEnd,
                           DictEnd], Ok(Dict(map)));
    }

    #[test]
    fn decodes_nested_dict() {
        let mut inner = TreeMap::new();
        inner.insert(Key::from_str("inner"), Number(2));
        let mut outer = TreeMap::new();
        outer.insert(Key::from_str("dict"), Dict(inner));
        outer.insert(Key::from_str("outer"), Number(1));
        assert_decoded_eq([DictStart,
                           DictKey(bytes("outer")),
                           NumberValue(1),
                           DictKey(bytes("dict")),
                           DictStart,
                           DictKey(bytes("inner")),
                           NumberValue(2),
                           DictEnd,
                           DictEnd], Ok(Dict(outer)));
    }

    #[test]
    fn decode_error_on_parse_error() {
        let err = Error{ pos: 1, msg: "error msg".to_string() };
        let perr = ParseError(err.clone());
        assert_decoded_eq([perr.clone()], Err(err.clone()));
        assert_decoded_eq([NumberValue(1), perr.clone()], Err(err.clone()));
        assert_decoded_eq([ListStart,
                           perr.clone()], Err(err.clone()));
        assert_decoded_eq([ListStart,
                           ByteStringValue(bytes("foo")),
                           perr.clone()], Err(err.clone()));
        assert_decoded_eq([DictStart,
                           perr.clone()], Err(err.clone()));
        assert_decoded_eq([DictStart,
                           DictKey(bytes("foo")),
                           perr.clone()], Err(err.clone()));
    }
}

#[cfg(test)]
mod bench {
    extern crate test;

    use self::test::Bencher;

    use std::io;

    use serialize::{Encodable, Decodable};

    use streaming::StreamingParser;
    use super::{Encoder, Decoder, Parser, DecoderResult};

    #[bench]
    fn encode_large_vec_of_uint(bh: &mut Bencher) {
        let v = Vec::from_fn(100, |n| n);
        bh.iter(|| {
            let mut w = io::MemWriter::with_capacity(v.len() * 10);
            {
                let mut enc = Encoder::new(&mut w);
                let _ = v.encode(&mut enc);
            }
            w.unwrap()
        });
        bh.bytes = v.len() as u64 * 4;
    }


    #[bench]
    fn decode_large_vec_of_uint(bh: &mut Bencher) {
        let v = Vec::from_fn(100, |n| n);
        let b = Encoder::buffer_encode(&v).unwrap();
        bh.iter(|| {
            let streaming_parser = StreamingParser::new(b.clone().move_iter());
            let mut parser = Parser::new(streaming_parser);
            let bencode = parser.parse().unwrap();
            let mut decoder = Decoder::new(&bencode);
            let result: DecoderResult<~[uint]> = Decodable::decode(&mut decoder);
            result
        });
        bh.bytes = b.len() as u64 * 4;
    }
}

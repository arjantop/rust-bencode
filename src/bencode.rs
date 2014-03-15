// Copyright 2014 Arjan Topolovec
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[crate_id = "bencode"];
#[license = "MIT/ASL2"];
#[crate_type = "rlib"];
#[crate_type = "dylib"];
#[deny(warnings)];
#[allow(deprecated_owned_vector)];
#[feature(macro_rules)];

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
      string: ~str,
      id: uint,
  }

  fn main() {
      let s = MyStruct { string: ~"Hello bencode", id: 1 };
      let result: ~[u8] = Encoder::buffer_encode(&s).unwrap();
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
      b: ~str,
      c: ~[u8],
  }

  impl ToBencode for MyStruct {
      fn to_bencode(&self) -> bencode::Bencode {
          let mut m = TreeMap::new();
          m.insert(Key::from_str("a"), self.a.to_bencode());
          m.insert(Key::from_str("b"), self.b.to_bencode());
          m.insert(Key::from_str("c"), bencode::ByteString(self.c.clone()));
          bencode::Dict(m)
      }
  }

  fn main() {
      let s = MyStruct{ a: 5, b: ~"foo", c: ~[1, 2, 3, 4] };
      let bencode: bencode::Bencode = s.to_bencode();
      let result: ~[u8] = bencode.to_bytes().unwrap();
  }

  ```

  # Decoding

  ## Using `Decodable`

  ```rust
  extern crate serialize;
  extern crate bencode;

  use serialize::{Encodable, Decodable};

  use bencode::{Encoder, Decoder};

  #[deriving(Encodable, Decodable, Eq)]
  struct MyStruct {
      a: int,
      b: ~str,
      c: ~[u8],
  }

  fn main() {
      let s = MyStruct{ a: 5, b: ~"foo", c: ~[1, 2, 3, 4] };
      let enc: ~[u8] = Encoder::buffer_encode(&s).unwrap();

      let bencode: bencode::Bencode = bencode::from_owned(enc).unwrap();
      let mut decoder = Decoder::new(&bencode);
      let result: MyStruct = Decodable::decode(&mut decoder);
      assert!(s == result)
  }
  ```  

  ## Using `FromBencode`

  ```rust
  extern crate collections;
  extern crate bencode;

  use collections::TreeMap;

  use bencode::{FromBencode, ToBencode, Dict, Key};

  #[deriving(Eq)]
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
      let enc: ~[u8] = s.to_bencode().to_bytes().unwrap();

      let bencode: bencode::Bencode = bencode::from_owned(enc).unwrap();
      let result: MyStruct = FromBencode::from_bencode(&bencode).unwrap();
      assert!(s == result)
  }
  ```

  ## Using Streaming Parser

  ```rust
  extern crate serialize;
  extern crate bencode;

  use bencode::{StreamingParser};
  use serialize::Encodable;

  use bencode::Encoder;

  #[deriving(Encodable, Decodable, Eq)]
  struct MyStruct {
      a: int,
      b: ~str,
      c: ~[u8],
  }

  fn main() {
      let s = MyStruct{ a: 5, b: ~"foo", c: ~[1, 2, 3, 4] };
      let enc: ~[u8] = Encoder::buffer_encode(&s).unwrap();

      let mut streaming = StreamingParser::new(enc.move_iter());
      for event in streaming {
          match event {
              bencode::DictStart => println!("dict start"),
              bencode::DictEnd => println!("dict end"),
              bencode::NumberValue(n) => println!("number = {}", n),
              // ...
              _ => println!("Unhandled event: {}", event)
          }
      }
  }
  ```
*/

extern crate collections;
extern crate serialize;

use std::io;
use std::io::{IoResult, IoError};
use std::fmt;
use std::str;
use std::str::raw;
use std::vec;
use std::vec_ng::Vec;

use serialize::{Encodable};

use collections::treemap::TreeMap;
use collections::hashmap::HashMap;

#[deriving(Eq, Clone)]
pub enum Bencode {
    Number(i64),
    ByteString(~[u8]),
    List(List),
    Dict(Dict),
}

impl fmt::Show for Bencode {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &Number(v) => write!(fmt.buf, "{}", v),
            &ByteString(ref v) => write!(fmt.buf, "s{}", v),
            &List(ref v) => write!(fmt.buf, "{}", v),
            &Dict(ref v) => {
                try!(write!(fmt.buf, r"\{"));
                let mut first = true;
                for (key, value) in v.iter() {
                    if first {
                        first = false;
                    } else {
                        try!(write!(fmt.buf, ", "));
                    }
                    try!(write!(fmt.buf, "{}: {}", *key, *value));
                }
                write!(fmt.buf, r"\}")
            }
        }
    }
}

#[deriving(Eq, Clone, TotalOrd, TotalEq, Ord, Show, Hash)]
pub struct Key(~[u8]);

pub type List = ~[Bencode];
pub type Dict = TreeMap<Key, Bencode>;

impl Bencode {
    pub fn to_writer(&self, writer: &mut io::Writer) -> io::IoResult<()> {
        let mut encoder = Encoder::new(writer);
        self.encode(&mut encoder);
        encoder.error
    }

    pub fn to_bytes(&self) -> io::IoResult<~[u8]> {
        let mut writer = io::MemWriter::new();
        match self.to_writer(&mut writer) {
            Ok(_) => Ok(writer.unwrap()),
            Err(err) => Err(err)
        }
    }
}

impl Key {
    pub fn from_str(s: &'static str) -> Key {
        Key(s.as_bytes().to_owned())
    }
}

impl<E: serialize::Encoder> Encodable<E> for Bencode {
    fn encode(&self, e: &mut E) {
        match self {
            &Number(v) => e.emit_i64(v),
            &ByteString(ref v) => e.emit_str(unsafe { raw::from_utf8(*v) }),
            &List(ref v) => v.encode(e),
            &Dict(ref v) => v.encode(e)
        }
    }
}

impl<E: serialize::Encoder> Encodable<E> for Key {
    fn encode(&self, e: &mut E) {
        let &Key(ref key) = self;
        e.emit_str(unsafe { raw::from_utf8(*key) })
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
        ByteString(~[])
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
        ByteString(std::f32::to_str_hex(*self).as_bytes().into_owned())
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
        ByteString(std::f64::to_str_hex(*self).as_bytes().into_owned())
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
            ByteString(bytes!("true").to_owned())
        } else {
            ByteString(bytes!("false").to_owned())
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
        ByteString(self.to_str().into_bytes())
    }
}

impl FromBencode for char {
    fn from_bencode(bencode: &Bencode) -> Option<char> {
        let s: Option<~str> = FromBencode::from_bencode(bencode);
        s.and_then(|s| {
            if s.char_len() == 1 {
                Some(s.char_at(0))
            } else {
                None
            }
        })
    }
}

impl ToBencode for ~str {
    fn to_bencode(&self) -> Bencode { ByteString(self.clone().into_bytes()) }
}

impl FromBencode for ~str {
    fn from_bencode(bencode: &Bencode) -> Option<~str> {
        match bencode {
            &ByteString(ref v) => std::str::from_utf8(*v).map(|s| s.to_owned()),
            _ => None
        }
    }
}

impl<T: ToBencode> ToBencode for ~[T] {
    fn to_bencode(&self) -> Bencode { List(self.map(|e| e.to_bencode())) }
}

impl<T: FromBencode> FromBencode for ~[T] {
    fn from_bencode(bencode: &Bencode) -> Option<~[T]> {
        match bencode {
            &List(ref es) => {
                let mut list = ~[];
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
            m.insert(Key(key.as_bytes().to_owned()), value.to_bencode());
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
                                Some(v) => m.insert(k.to_owned(), v),
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

impl<T: ToBencode> ToBencode for TreeMap<~str, T> {
    fn to_bencode(&self) -> Bencode {
        map_to_bencode!(self)
    }
}

impl<T: FromBencode> FromBencode for TreeMap<~str, T> {
    fn from_bencode(bencode: &Bencode) -> Option<TreeMap<~str, T>> {
        map_from_bencode!(TreeMap)
    }
}

impl<T: ToBencode> ToBencode for HashMap<~str, T> {
    fn to_bencode(&self) -> Bencode {
        map_to_bencode!(self)
    }
}

impl<T: FromBencode> FromBencode for HashMap<~str, T> {
    fn from_bencode(bencode: &Bencode) -> Option<HashMap<~str, T>> {
        map_from_bencode!(HashMap)
    }
}

pub fn from_buffer(buf: &[u8]) -> Result<Bencode, Error> {
    from_owned(buf.to_owned())
}

pub fn from_owned(buf: ~[u8]) -> Result<Bencode, Error> {
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

pub struct Encoder<'a> {
    priv writer: &'a mut io::Writer,
    priv writers: Vec<io::MemWriter>,
    priv expect_key: bool,
    priv keys: Vec<Key>,
    priv error: io::IoResult<()>,
    priv stack: Vec<TreeMap<Key, ~[u8]>>,
}

impl<'a> Encoder<'a> {
    pub fn new(writer: &'a mut io::Writer) -> Encoder<'a> {
        Encoder {
            writer: writer,
            writers: Vec::new(),
            expect_key: false,
            keys: Vec::new(),
            error: Ok(()),
            stack: Vec::new()
        }
    }

    pub fn buffer_encode<T: Encodable<Encoder<'a>>>(val: &T) -> IoResult<~[u8]> {
        let mut writer = io::MemWriter::new();
        {
            let mut encoder = Encoder::new(&mut writer);
            val.encode(&mut encoder);
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

    fn encode_dict(&mut self, dict: &TreeMap<Key, ~[u8]>) {
        tryenc!(write!(self.get_writer(), "d"));
        for (key, value) in dict.iter() {
            key.encode(self);
            tryenc!(self.get_writer().write(value.as_slice()));
        }
        tryenc!(write!(self.get_writer(), "e"));
    }

    fn error(&mut self, msg: &'static str) {
        self.error = Err(IoError {
            kind: io::InvalidInput,
            desc: msg,
            detail: None
        });
        fail!(msg)
    }
}

macro_rules! expect_value(() => {
    if self.expect_key {
        self.error("Only 'string' map keys allowed");
    }
})

impl<'a> serialize::Encoder for Encoder<'a> {
    fn emit_nil(&mut self) { expect_value!(); tryenc!(write!(self.get_writer(), "0:")) }

    fn emit_uint(&mut self, v: uint) { self.emit_i64(v as i64); }

    fn emit_u8(&mut self, v: u8) { self.emit_i64(v as i64); }

    fn emit_u16(&mut self, v: u16) { self.emit_i64(v as i64); }

    fn emit_u32(&mut self, v: u32) { self.emit_i64(v as i64); }

    fn emit_u64(&mut self, v: u64) { self.emit_i64(v as i64); }

    fn emit_int(&mut self, v: int) { self.emit_i64(v as i64); }

    fn emit_i8(&mut self, v: i8) { self.emit_i64(v as i64); }

    fn emit_i16(&mut self, v: i16) { self.emit_i64(v as i64); }

    fn emit_i32(&mut self, v: i32) { self.emit_i64(v as i64); }

    fn emit_i64(&mut self, v: i64) { expect_value!(); tryenc!(write!(self.get_writer(), "i{}e", v)) }

    fn emit_bool(&mut self, v: bool) {
        expect_value!(); 
        if v {
            self.emit_str("true");
        } else {
            self.emit_str("false");
        }
    }

    fn emit_f32(&mut self, v: f32) {
        expect_value!(); 
        self.emit_str(std::f32::to_str_hex(v));
    }

    fn emit_f64(&mut self, v: f64) {
        expect_value!(); 
        self.emit_str(std::f64::to_str_hex(v));
    }

    fn emit_char(&mut self, v: char) { expect_value!(); self.emit_str(str::from_char(v)); }

    fn emit_str(&mut self, v: &str) {
        if self.expect_key {
            self.keys.push(Key(v.as_bytes().to_owned()));
        } else {
            tryenc!(write!(self.get_writer(), "{}:", v.len()));
            tryenc!(self.get_writer().write(v.as_bytes()));
        }
    }

    fn emit_enum(&mut self, _name: &str, _f: |&mut Encoder<'a>|) { unimplemented!(); }
    fn emit_enum_variant(&mut self, _v_name: &str, _v_id: uint, _len: uint, _f: |&mut Encoder<'a>|) { unimplemented!(); }
    fn emit_enum_variant_arg(&mut self, _a_idx: uint, _f: |&mut Encoder<'a>|) { unimplemented!(); }
    fn emit_enum_struct_variant(&mut self, _v_name: &str, _v_id: uint, _len: uint, _f: |&mut Encoder<'a>|) { unimplemented!(); }
    fn emit_enum_struct_variant_field(&mut self, _f_name: &str, _f_idx: uint, _f: |&mut Encoder<'a>|) { unimplemented!(); }

    fn emit_struct(&mut self, _name: &str, _len: uint, f: |&mut Encoder<'a>|) {
        expect_value!(); 
        self.stack.push(TreeMap::new());
        f(self);
        let dict = self.stack.pop().unwrap();
        self.encode_dict(&dict);
    }

    fn emit_struct_field(&mut self, f_name: &str, _f_idx: uint, f: |&mut Encoder<'a>|) {
        expect_value!(); 
        self.writers.push(io::MemWriter::new());
        f(self);
        let data = self.writers.pop().unwrap();
        let dict = self.stack.mut_last().unwrap();
        dict.insert(Key(f_name.as_bytes().to_owned()), data.unwrap());
    }

    fn emit_tuple(&mut self, _len: uint, _f: |&mut Encoder<'a>|) { unimplemented!(); }
    fn emit_tuple_arg(&mut self, _idx: uint, _f: |&mut Encoder<'a>|) { unimplemented!(); }
    fn emit_tuple_struct(&mut self, _name: &str, _len: uint, _f: |&mut Encoder<'a>|) { unimplemented!(); }
    fn emit_tuple_struct_arg(&mut self, _f_idx: uint, _f: |&mut Encoder<'a>|) { unimplemented!(); }
    fn emit_option(&mut self, _f: |&mut Encoder<'a>|) { unimplemented!(); }
    fn emit_option_none(&mut self) { unimplemented!(); }
    fn emit_option_some(&mut self, _f: |&mut Encoder<'a>|) { unimplemented!(); }

    fn emit_seq(&mut self, _len: uint, f: |this: &mut Encoder<'a>|) {
        expect_value!(); 
        tryenc!(write!(self.get_writer(), "l"));
        f(self);
        tryenc!(write!(self.get_writer(), "e"));
    }

    fn emit_seq_elt(&mut self, _idx: uint, f: |this: &mut Encoder<'a>|) {
        expect_value!(); 
        f(self);
    }

    fn emit_map(&mut self, _len: uint, f: |&mut Encoder<'a>|) {
        expect_value!(); 
        self.stack.push(TreeMap::new());
        f(self);
        let dict = self.stack.pop().unwrap();
        self.encode_dict(&dict);
    }

    fn emit_map_elt_key(&mut self, _idx: uint, f: |&mut Encoder<'a>|) {
        expect_value!(); 
        self.writers.push(io::MemWriter::new());
        self.expect_key = true;
        f(self);
        self.expect_key = false;
    }

    fn emit_map_elt_val(&mut self, _idx: uint, f: |&mut Encoder<'a>|) {
        expect_value!(); 
        f(self);
        let key = self.keys.pop();
        let data = self.writers.pop().unwrap();
        let dict = self.stack.mut_last().unwrap();
        dict.insert(key.unwrap(), data.unwrap());
    }
}

#[deriving(Show, Eq, Clone)]
pub enum BencodeEvent {
    NumberValue(i64),
    ByteStringValue(~[u8]),
    ListStart,
    ListEnd,
    DictStart,
    DictKey(~[u8]),
    DictEnd,
    ParseError(Error),
}

#[deriving(Show, Eq, Clone)]
pub struct Error {
    pos: u32,
    msg: ~str,
}

#[deriving(Show, Eq, Clone)]
enum BencodePosition {
    ListPosition,
    KeyPosition,
    ValuePosition
}

pub struct StreamingParser<T> {
    priv reader: T,
    priv pos: u32,
    priv decoded: u32,
    priv end: bool,
    priv curr: Option<u8>,
    priv stack: Vec<BencodePosition>,
}

macro_rules! expect(($ch:pat, $ex:expr) => (
    match self.curr_char() {
        Some($ch) => self.next_byte(),
        _ => return self.error($ex)
    }
))

macro_rules! check_nesting(() => (
    if self.decoded > 0 && self.stack.len() == 0 {
        return self.error_msg(~"Only one value allowed outside of containers")
    }
))

impl<T: Iterator<u8>> StreamingParser<T> {
    pub fn new(mut reader: T) -> StreamingParser<T> {
        let next = reader.next();
        StreamingParser {
            reader: reader,
            pos: 0,
            decoded: 0,
            end: false,
            curr: next,
            stack: Vec::new()
        }
    }

    fn next_byte(&mut self) {
        self.pos += 1;
        self.curr = self.reader.next();
        self.end = self.curr.is_none();
    }
    
    fn next_bytes(&mut self, len: uint) -> Result<~[u8], Error> {
        let mut bytes = vec::with_capacity(len);
        for _ in range(0, len) {
            match self.curr {
                Some(x) =>  bytes.push(x),
                None => {
                    let msg = format!("Expecting {} bytes but only got {}", len, bytes.len());
                    return self.error_msg(msg)
                }
            }
            self.next_byte();
        }
        Ok(bytes)
    }

    #[inline(always)]
    fn curr_char(&mut self) -> Option<char> {
        self.curr.map(|v| v as char)
    }

    fn error<T>(&mut self, expected: ~str) -> Result<T, Error> {
        let got = self.curr_char();
        let got_char = match got {
            Some(x) => alphanum_to_str(x),
            None => ~"EOF"
        };
        self.error_msg(format!("Expecting '{}' but got '{}'", expected, got_char))
    }

    fn error_msg<T>(&mut self, msg: ~str) -> Result<T, Error> {
        Err(Error {
            pos: self.pos,
            msg: msg
        })
    }

    #[inline(always)]
    fn parse_number(&mut self) -> Result<i64, Error> {
        let sign = match self.curr_char() {
            Some('-') => {
                self.next_byte();
                -1
            }
            _ => 1
        };
        let num = try!(self.parse_number_unsigned());
        expect!('e', ~"e");
        Ok(sign * num)
    }

    #[inline(always)]
    fn parse_number_unsigned(&mut self) -> Result<i64, Error> {
        let mut num = 0;
        match self.curr_char() {
            Some('0') => self.next_byte(),
            Some('1' .. '9') => {
                loop {
                    match self.curr_char() {
                        Some(ch @ '0' .. '9') => self.parse_digit(ch, &mut num),
                        _ => break
                    }
                    self.next_byte();
                }
            }
            _ => return self.error(~"0-9")
        };
        Ok(num)
    }

    #[inline(always)]
    fn parse_digit(&self, ch: char, num: &mut i64) {
        *num *= 10;
        *num += ch.to_digit(10).unwrap() as i64;
    }

    #[inline(always)]
    fn parse_bytestring(&mut self) -> Result<~[u8], Error> {
        let len = try!(self.parse_number_unsigned());
        expect!(':', ~":");
        let bytes = try!(self.next_bytes(len as uint));
        Ok(bytes)
    }

    #[inline(always)]
    fn parse_end(&mut self) -> Result<BencodeEvent, Error> {
        self.next_byte();
        match self.stack.pop() {
            Some(ListPosition) => Ok(ListEnd),
            Some(ValuePosition) => {
                Ok(DictEnd)
            }
            _ => return self.error_msg(~"Unmatched value ending")
        }
    }

    #[inline(always)]
    fn parse_key(&mut self) -> Result<BencodeEvent, Error> {
        check_nesting!();
        match self.curr_char() {
            Some('0' .. '9') => {
                self.decoded += 1;
                let res = try!(self.parse_bytestring());
                Ok(DictKey(res))
            }
            Some('e') => self.parse_end(),
            _ => self.error(~"0-9 or e")
        }
    }

    #[inline(always)]
    fn parse_event(&mut self) -> Result<BencodeEvent, Error> {
        match self.curr_char() {
            Some('i') => {
                check_nesting!();
                self.next_byte();
                self.decoded += 1;
                let res = try!(self.parse_number());
                Ok(NumberValue(res))
            }
            Some('0' .. '9') => {
                check_nesting!();
                self.decoded += 1;
                let res = try!(self.parse_bytestring());
                Ok(ByteStringValue(res))
            }
            Some('l') => {
                check_nesting!();
                self.next_byte();
                self.decoded += 1;
                self.stack.push(ListPosition);
                Ok(ListStart)
            }
            Some('d') => {
                check_nesting!();
                self.next_byte();
                self.decoded += 1;
                self.stack.push(KeyPosition);
                Ok(DictStart)
            }
            Some('e') => self.parse_end(),
            _ => self.error(~"i or 0-9 or l or d or e")
        }
    }
}

impl<T: Iterator<u8>> Iterator<BencodeEvent> for StreamingParser<T> {
    #[inline(always)]
    fn next(&mut self) -> Option<BencodeEvent> {
        if self.end {
            return None
        }
        let result = match self.stack.pop() {
            Some(KeyPosition) => {
                self.stack.push(ValuePosition);
                self.parse_key()
            }
            pos => {
                match pos {
                    Some(ValuePosition) => self.stack.push(KeyPosition),
                    Some(_) => self.stack.push(pos.unwrap()),
                    None => {}
                }
                self.parse_event()
            }
        };
        match result {
            Ok(ev) => Some(ev),
            Err(err) => {
                self.end = true;
                Some(ParseError(err))
            }
        }
    }
}

fn alphanum_to_str(ch: char) -> ~str {
    if ch.is_alphanumeric() {
        ch.to_str()
    } else {
        (ch as u8).to_str()
    }
}

pub struct Parser<T> {
    priv reader: T,
    priv depth: u32,
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
        let mut list = ~[];
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

pub struct Decoder<'a> {
    priv keys: Vec<Key>,
    priv expect_key: bool,
    priv stack: Vec<&'a Bencode>,
}

impl<'a> Decoder<'a> {
    pub fn new(bencode: &'a Bencode) -> Decoder<'a> {
        Decoder {
            keys: Vec::new(),
            expect_key: false,
            stack: Vec::from_slice([bencode])
        }
    }

    fn try_read<T: FromBencode>(&mut self) -> T {
        self.stack.pop().and_then(|b| FromBencode::from_bencode(b)).unwrap()
    }

    fn error(&self, msg: &'static str) -> ! {
        fail!(msg)
    }
}

impl<'a> serialize::Decoder for Decoder<'a> {
    fn read_nil(&mut self) { expect_value!(); self.try_read() }

    fn read_uint(&mut self) -> uint { expect_value!(); self.try_read() }

    fn read_u8(&mut self) -> u8 { expect_value!(); self.try_read() }

    fn read_u16(&mut self) -> u16 { expect_value!(); self.try_read() }

    fn read_u32(&mut self) -> u32 { expect_value!(); self.try_read() }

    fn read_u64(&mut self) -> u64 { expect_value!(); self.try_read() }

    fn read_int(&mut self) -> int { expect_value!(); self.try_read() }

    fn read_i8(&mut self) -> i8 { expect_value!(); self.try_read() }

    fn read_i16(&mut self) -> i16 { expect_value!(); self.try_read() }

    fn read_i32(&mut self) -> i32 { expect_value!(); self.try_read() }

    fn read_i64(&mut self) -> i64 { expect_value!(); self.try_read() } 

    fn read_bool(&mut self) -> bool { expect_value!(); self.try_read() }

    fn read_f32(&mut self) -> f32 { expect_value!(); self.try_read() }

    fn read_f64(&mut self) -> f64 { expect_value!(); self.try_read() }

    fn read_char(&mut self) -> char { expect_value!(); self.try_read() }

    fn read_str(&mut self) -> ~str {
        if self.expect_key {
            let Key(b) = self.keys.pop().unwrap();
            str::from_utf8_owned(b).unwrap()
        } else {
            self.try_read()
        }
    }

    fn read_enum<T>(&mut self, _name: &str, _f: |&mut Decoder<'a>| -> T) -> T { unimplemented!(); }
    fn read_enum_variant<T>(&mut self, _names: &[&str], _f: |&mut Decoder<'a>, uint| -> T) -> T { unimplemented!(); }
    fn read_enum_variant_arg<T>(&mut self, _a_idx: uint, _f: |&mut Decoder<'a>| -> T) -> T { unimplemented!(); }
    fn read_enum_struct_variant<T>(&mut self, _names: &[&str], _f: |&mut Decoder<'a>, uint| -> T) -> T { unimplemented!(); }
    fn read_enum_struct_variant_field<T>(&mut self, _f_name: &str, _f_idx: uint, _f: |&mut Decoder<'a>| -> T) -> T { unimplemented!(); }

    fn read_struct<T>(&mut self, _s_name: &str, _len: uint, f: |&mut Decoder<'a>| -> T) -> T {
        expect_value!(); 
        let res = f(self);
        self.stack.pop();
        res
    }

    fn read_struct_field<T>(&mut self, f_name: &str, _f_idx: uint, f: |&mut Decoder<'a>| -> T) -> T {
        expect_value!(); 
        let val = match self.stack.last() {
            Some(v) => {
                match *v {
                    &Dict(ref m) => {
                        match m.find(&Key(f_name.as_bytes().to_owned())) {
                            Some(v) => v,
                            None => fail!()
                        }
                    }
                    _ => fail!()
                }
            }
            _ => fail!()
        };
        self.stack.push(val);
        f(self)
    }

    fn read_tuple<T>(&mut self, _f: |&mut Decoder<'a>, uint| -> T) -> T { unimplemented!(); }
    fn read_tuple_arg<T>(&mut self, _a_idx: uint, _f: |&mut Decoder<'a>| -> T) -> T { unimplemented!(); }
    fn read_tuple_struct<T>(&mut self, _s_name: &str, _f: |&mut Decoder<'a>, uint| -> T) -> T { unimplemented!(); }
    fn read_tuple_struct_arg<T>(&mut self, _a_idx: uint, _f: |&mut Decoder<'a>| -> T) -> T { unimplemented!(); }
    fn read_option<T>(&mut self, _f: |&mut Decoder<'a>, bool| -> T) -> T { unimplemented!(); }

    fn read_seq<T>(&mut self, f: |&mut Decoder<'a>, uint| -> T) -> T {
        expect_value!(); 
        let len = match self.stack.pop() {
            Some(&List(ref list)) => {
                for v in list.rev_iter() {
                    self.stack.push(v);
                }
                list.len()
            }
            _ => fail!()
        };
        f(self, len)
    }

    fn read_seq_elt<T>(&mut self, _idx: uint, f: |&mut Decoder<'a>| -> T) -> T { expect_value!(); f(self) }

    fn read_map<T>(&mut self, f: |&mut Decoder<'a>, uint| -> T) -> T {
        expect_value!(); 
        let len = match self.stack.pop() {
            Some(&Dict(ref m)) => {
                for (key, value) in m.iter() {
                    self.keys.push(key.clone());
                    self.stack.push(value);
                }
                m.len()
            }
            _ => fail!()
        };
        f(self, len)
    }

    fn read_map_elt_key<T>(&mut self, _idx: uint, f: |&mut Decoder<'a>| -> T) -> T {
        expect_value!(); 
        self.expect_key = true;
        let res = f(self);
        self.expect_key = false;
        res
    }

    fn read_map_elt_val<T>(&mut self, _idx: uint, f: |&mut Decoder<'a>| -> T) -> T {
        expect_value!(); 
        f(self)
    }
}

#[cfg(test)]
mod tests {
    use std::str::raw;
    use serialize::{Encodable, Decodable};
    use collections::treemap::TreeMap;
    use collections::hashmap::HashMap;

    use super::{Bencode, ToBencode, Error};
    use super::{Encoder, ByteString, List, Number, Dict, Key};
    use super::{StreamingParser, BencodeEvent, NumberValue, ByteStringValue,
                ListStart, ListEnd, DictStart, DictKey, DictEnd, ParseError};
    use super::{Parser, Decoder};
    use super::alphanum_to_str;

    macro_rules! assert_encoding(($value:expr, $expected:expr) => ({
        let value = $value;
        let encoded = match Encoder::buffer_encode(&value) {
            Ok(e) => e,
            Err(err) => fail!("Unexpected failure: {}", err)
        };
        assert_eq!($expected, encoded);
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
        let bencode = super::from_owned(encoded).unwrap();
        let mut decoder = Decoder::new(&bencode);
        let result = Decodable::decode(&mut decoder);
        assert_eq!(value, result);
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

    fn bytes(s: &str) -> ~[u8] {
        s.as_bytes().to_owned()
    }

    gen_complete_test!(encodes_unit,
                       tobencode_unit,
                       identity_unit,
                       () -> bytes("0:"))

    gen_complete_test!(encodes_zero_int,
                       tobencode_zero_int,
                       identity_zero_int,
                       0i -> bytes("i0e"))

    gen_complete_test!(encodes_positive_int,
                       tobencode_positive_int,
                       identity_positive_int,
                       5i -> bytes("i5e"),
                       99i -> bytes("i99e"),
                       ::std::int::MAX -> bytes(format!("i{}e", ::std::int::MAX)))

    gen_complete_test!(encodes_negative_int,
                       tobencode_negative_int,
                       identity_negative_int,
                       -5i -> bytes("i-5e"),
                       -99i -> bytes("i-99e"),
                       ::std::int::MIN -> bytes(format!("i{}e", ::std::int::MIN)))

    gen_complete_test!(encodes_zero_i8,
                       tobencode_zero_i8,
                       identity_zero_i8,
                       0i8 -> bytes("i0e"))

    gen_complete_test!(encodes_positive_i8,
                       tobencode_positive_i8,
                       identity_positive_i8,
                       5i8 -> bytes("i5e"),
                       99i8 -> bytes("i99e"),
                       ::std::i8::MAX -> bytes(format!("i{}e", ::std::i8::MAX)))

    gen_complete_test!(encodes_negative_i8,
                       tobencode_negative_i8,
                       identity_negative_i8,
                       -5i8 -> bytes("i-5e"),
                       -99i8 -> bytes("i-99e"),
                       ::std::i8::MIN -> bytes(format!("i{}e", ::std::i8::MIN)))

    gen_complete_test!(encodes_zero_i16,
                       tobencode_zero_i16,
                       identity_zero_i16,
                       0i16 -> bytes("i0e"))

    gen_complete_test!(encodes_positive_i16,
                       tobencode_positive_i16,
                       identity_positive_i16,
                       5i16 -> bytes("i5e"),
                       99i16 -> bytes("i99e"),
                       ::std::i16::MAX -> bytes(format!("i{}e", ::std::i16::MAX)))

    gen_complete_test!(encodes_negative_i16,
                       tobencode_negative_i16,
                       identity_negative_i16,
                       -5i16 -> bytes("i-5e"),
                       -99i16 -> bytes("i-99e"),
                       ::std::i16::MIN -> bytes(format!("i{}e", ::std::i16::MIN)))

    gen_complete_test!(encodes_zero_i32,
                       tobencode_zero_i32,
                       identity_zero_i32,
                       0i32 -> bytes("i0e"))

    gen_complete_test!(encodes_positive_i32,
                       tobencode_positive_i32,
                       identity_positive_i32,
                       5i32 -> bytes("i5e"),
                       99i32 -> bytes("i99e"),
                       ::std::i32::MAX -> bytes(format!("i{}e", ::std::i32::MAX)))

    gen_complete_test!(encodes_negative_i32,
                       tobencode_negative_i32,
                       identity_negative_i32,
                       -5i32 -> bytes("i-5e"),
                       -99i32 -> bytes("i-99e"),
                       ::std::i32::MIN -> bytes(format!("i{}e", ::std::i32::MIN)))

    gen_complete_test!(encodes_zero_i64,
                       tobencode_zero_i64,
                       identity_zero_i64,
                       0i64 -> bytes("i0e"))

    gen_complete_test!(encodes_positive_i64,
                       tobencode_positive_i64,
                       identity_positive_i64,
                       5i64 -> bytes("i5e"),
                       99i64 -> bytes("i99e"),
                       ::std::i64::MAX -> bytes(format!("i{}e", ::std::i64::MAX)))

    gen_complete_test!(encodes_negative_i64,
                       tobencode_negative_i64,
                       identity_negative_i64,
                       -5i64 -> bytes("i-5e"),
                       -99i64 -> bytes("i-99e"),
                       ::std::i64::MIN -> bytes(format!("i{}e", ::std::i64::MIN)))

    gen_complete_test!(encodes_zero_uint,
                       tobencode_zero_uint,
                       identity_zero_uint,
                       0u -> bytes("i0e"))

    gen_complete_test!(encodes_positive_uint,
                       tobencode_positive_uint,
                       identity_positive_uint,
                       5u -> bytes("i5e"),
                       99u -> bytes("i99e"),
                       ::std::uint::MAX / 2 -> bytes(format!("i{}e", ::std::uint::MAX / 2)))

    gen_complete_test!(encodes_zero_u8,
                       tobencode_zero_u8,
                       identity_zero_u8,
                       0u8 -> bytes("i0e"))

    gen_complete_test!(encodes_positive_u8,
                       tobencode_positive_u8,
                       identity_positive_u8,
                       5u8 -> bytes("i5e"),
                       99u8 -> bytes("i99e"),
                       ::std::u8::MAX -> bytes(format!("i{}e", ::std::u8::MAX)))

    gen_complete_test!(encodes_zero_u16,
                       tobencode_zero_u16,
                       identity_zero_u16,
                       0u16 -> bytes("i0e"))

    gen_complete_test!(encodes_positive_u16,
                       tobencode_positive_u16,
                       identity_positive_u16,
                       5u16 -> bytes("i5e"),
                       99u16 -> bytes("i99e"),
                       ::std::u16::MAX -> bytes(format!("i{}e", ::std::u16::MAX)))

    gen_complete_test!(encodes_zero_u32,
                       tobencode_zero_u32,
                       identity_zero_u32,
                       0u32 -> bytes("i0e"))

    gen_complete_test!(encodes_positive_u32,
                       tobencode_positive_u32,
                       identity_positive_u32,
                       5u32 -> bytes("i5e"),
                       99u32 -> bytes("i99e"),
                       ::std::u32::MAX -> bytes(format!("i{}e", ::std::u32::MAX)))

    gen_complete_test!(encodes_zero_u64,
                       tobencode_zero_u64,
                       identity_zero_u64,
                       0u64 -> bytes("i0e"))

    gen_complete_test!(encodes_positive_u64,
                       tobencode_positive_u64,
                       identity_positive_u64,
                       5u64 -> bytes("i5e"),
                       99u64 -> bytes("i99e"),
                       ::std::u64::MAX / 2 -> bytes(format!("i{}e", ::std::u64::MAX / 2)))

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
                      ~"" -> bytes("0:"))

    gen_complete_test!(encode_str,
                      tobencode_str,
                      identity_str,
                      ~"a" -> bytes("1:a"),
                      ~"foo" -> bytes("3:foo"),
                      ~"This is nice!?#$%" -> bytes("17:This is nice!?#$%"))

    gen_complete_test!(encode_str_with_multibyte_chars,
                      tobencode_str_with_multibyte_chars,
                      identity_str_with_multibyte_chars,
                      ~"Löwe 老虎 Léopard" -> bytes("21:Löwe 老虎 Léopard"),
                      ~"いろはにほへとちりぬるを" -> bytes("36:いろはにほへとちりぬるを"))

    gen_complete_test!(encodes_empty_vec,
                       tobencode_empty_vec,
                       identity_empty_vec,
                       {
                           let empty: ~[u8] = ~[];
                           empty
                       } -> bytes("le"))

    gen_complete_test!(encodes_nonmpty_vec,
                       tobencode_nonmpty_vec,
                       identity_nonmpty_vec,
                       ~[0, 1, 3, 4] -> bytes("li0ei1ei3ei4ee"),
                       ~[~"foo", ~"b"] -> bytes("l3:foo1:be"))

    gen_complete_test!(encodes_nested_vec,
                       tobencode_nested_vec,
                       identity_nested_vec,
                       ~[~[1], ~[2, 3], ~[]] -> bytes("lli1eeli2ei3eelee"))

    #[deriving(Eq, Show, Encodable, Decodable)]
    struct SimpleStruct {
        a: uint,
        b: ~[~str],
    }

    #[deriving(Eq, Show, Encodable, Decodable)]
    struct InnerStruct {
        field_one: (),
        list: ~[uint],
        abc: ~str
    }

    #[deriving(Eq, Show, Encodable, Decodable)]
    struct OuterStruct {
        inner: ~[InnerStruct],
        is_true: bool
    }

    gen_encode_identity_test!(encodes_struct,
                              identity_struct,
                              SimpleStruct {
                                  b: ~[~"foo", ~"baar"],
                                  a: 123
                              } -> bytes("d1:ai123e1:bl3:foo4:baaree"),
                              SimpleStruct {
                                  a: 1234567890,
                                  b: ~[]
                              } -> bytes("d1:ai1234567890e1:blee"))

    gen_encode_identity_test!(encodes_nested_struct,
                              identity_nested_Struct,
                              OuterStruct {
                                  is_true: true,
                                  inner: ~[InnerStruct {
                                      field_one: (),
                                      list: ~[99, 5],
                                      abc: ~"rust"
                                  }, InnerStruct {
                                      field_one: (),
                                      list: ~[],
                                      abc: ~""
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
                       map!(HashMap, (~"a", 1)) -> bytes("d1:ai1ee"),
                       map!(HashMap, (~"foo", ~"a"), (~"bar", ~"bb")) -> bytes("d3:bar2:bb3:foo1:ae"))

    gen_complete_test!(encodes_nested_hashmap,
                       bencode_nested_hashmap,
                       identity_nested_hashmap,
                       map!(HashMap, (~"a", map!(HashMap, (~"foo", 101), (~"bar", 102)))) -> bytes("d1:ad3:bari102e3:fooi101eee"))
    #[test]
    #[should_fail]
    fn decode_error_on_wrong_map_key_type() {
        let benc = Dict(map!(TreeMap, (Key(bytes("foo")), ByteString(bytes("bar")))));
        let mut decoder = Decoder::new(&benc);
        let _res: TreeMap<int, ~str> = Decodable::decode(&mut decoder);
    }

    #[test]
    #[should_fail]
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

    fn try_bencode(bencode: Bencode) -> ~[u8] {
        match bencode.to_bytes() {
            Ok(v) => v,
            Err(err) => fail!("Unexpected error: {}", err)
        }
    }

    #[test]
    fn encodes_empty_bytestring() {
        assert_eq!(try_bencode(ByteString(~[])), bytes("0:"));
    }

    #[test]
    fn encodes_nonempty_bytestring() {
        assert_eq!(try_bencode(ByteString((~"abc").into_bytes())), bytes("3:abc"));
        assert_eq!(try_bencode(ByteString(~[0, 1, 2, 3])), bytes("4:") + ~[0u8, 1, 2, 3]);
    }

    #[test]
    fn encodes_empty_list() {
        assert_eq!(try_bencode(List(~[])), bytes("le"));
    }

    #[test]
    fn encodes_nonempty_list() {
        assert_eq!(try_bencode(List(~[Number(1)])), bytes("li1ee"));
        assert_eq!(try_bencode(List(~[ByteString((~"foobar").into_bytes()),
                          Number(-1)])), bytes("l6:foobari-1ee"));
    }

    #[test]
    fn encodes_nested_list() {
        assert_eq!(try_bencode(List(~[List(~[])])), bytes("llee"));
        let list = List(~[Number(1988), List(~[Number(2014)])]);
        assert_eq!(try_bencode(list), bytes("li1988eli2014eee"));
    }

    #[test]
    fn encodes_empty_dict() {
        assert_eq!(try_bencode(Dict(TreeMap::new())), bytes("de"));
    }

    #[test]
    fn encodes_dict_with_items() {
        let mut m = TreeMap::new();
        m.insert(Key((~"k1").into_bytes()), Number(1));
        assert_eq!(try_bencode(Dict(m.clone())), bytes("d2:k1i1ee"));
        m.insert(Key((~"k2").into_bytes()), ByteString(~[0, 0]));
        assert_eq!(try_bencode(Dict(m.clone())), bytes("d2:k1i1e2:k22:\0\0e"));
    }

    #[test]
    fn encodes_nested_dict() {
        let mut outer = TreeMap::new();
        let mut inner = TreeMap::new();
        inner.insert(Key((~"val").into_bytes()), ByteString(~[68, 0, 90]));
        outer.insert(Key((~"inner").into_bytes()), Dict(inner));
        assert_eq!(try_bencode(Dict(outer)), bytes("d5:innerd3:val3:D\0Zee"));
    }

    #[test]
    fn encodes_dict_fields_in_sorted_order() {
        let mut m = TreeMap::new();
        m.insert(Key((~"z").into_bytes()), Number(1));
        m.insert(Key((~"abd").into_bytes()), Number(3));
        m.insert(Key((~"abc").into_bytes()), Number(2));
        assert_eq!(try_bencode(Dict(m)), bytes("d3:abci2e3:abdi3e1:zi1ee"));
    }

    fn assert_stream_eq(encoded: &str, expected: &[BencodeEvent]) {
        let mut parser = StreamingParser::new(encoded.bytes());
        let result = parser.to_owned_vec();
        assert_eq!(expected, result.as_slice());
    }

    #[test]
    fn parse_error_on_invalid_first_character() {
        for n in range(::std::u8::MIN, ::std::u8::MAX) {
            match n as char {
                'i' | '0' .. '9' | 'l' | 'd' | 'e' => continue,
                _ => {}
            };
            let msg = format!("Expecting 'i or 0-9 or l or d or e' but got '{}'", alphanum_to_str(n as char));
            assert_stream_eq(unsafe { raw::from_utf8([n]) },
                            [ParseError(Error{
                                pos: 0,
                                msg: msg })]);
        }
    }

    #[test]
    fn parses_number_zero() {
        assert_stream_eq("i0e", [NumberValue(0)]);
    }

    #[test]
    fn parses_positive_numbers() {
        assert_stream_eq("i5e", [NumberValue(5)]);
        assert_stream_eq(format!("i{}e", ::std::i64::MAX), [NumberValue(::std::i64::MAX)]);
    }

    #[test]
    fn parses_negative_numbers() {
        assert_stream_eq("i-5e", [NumberValue(-5)]);
        assert_stream_eq(format!("i{}e", ::std::i64::MIN), [NumberValue(::std::i64::MIN)]);
    }

    #[test]
    fn parse_error_on_number_without_ending() {
        assert_stream_eq("i10", [ParseError(Error{ pos: 3, msg: ~"Expecting 'e' but got 'EOF'" })]);
    }

    #[test]
    fn parse_error_on_number_with_leading_zero() {
        assert_stream_eq("i0215e", [ParseError(Error{ pos: 2, msg: ~"Expecting 'e' but got '2'" })]);
    }

    #[test]
    fn parse_error_on_empty_number() {
        assert_stream_eq("ie", [ParseError(Error{ pos: 1, msg: ~"Expecting '0-9' but got 'e'" })]);
    }

    #[test]
    fn parse_error_on_more_than_one_value_outside_of_containers() {
        let msg = ~"Only one value allowed outside of containers";
        assert_stream_eq("i1ei2e", [NumberValue(1), ParseError(Error{ pos: 3, msg: msg.clone() })]);
        assert_stream_eq("i10eli2ee", [NumberValue(10), ParseError(Error{ pos: 4, msg: msg.clone() })]);
        assert_stream_eq("1:ade", [ByteStringValue(bytes("a")), ParseError(Error{ pos: 3, msg: msg.clone() })]);
        assert_stream_eq("3:foo3:bar", [ByteStringValue(bytes("foo")),
                                        ParseError(Error{ pos: 5, msg: msg.clone() })]);
    }

    #[test]
    fn parses_empty_bytestring() {
        assert_stream_eq("0:", [ByteStringValue(~[])]);
    }

    #[test]
    fn parses_short_bytestring() {
        assert_stream_eq("6:abcdef", [ByteStringValue(bytes("abcdef"))]);
    }

    #[test]
    fn parses_long_bytestring() {
        let long = "baz".repeat(10);
        assert_stream_eq(format!("{}:{}", long.len(), long), [ByteStringValue(long.as_bytes().to_owned())]);
    }

    #[test]
    fn parse_error_on_too_short_data() {
        assert_stream_eq("5:abcd", [ParseError(Error { pos: 6, msg: ~"Expecting 5 bytes but only got 4" })]);
    }

    #[test]
    fn parse_error_on_malformed_bytestring() {
        assert_stream_eq("3abc", [ParseError(Error { pos: 1, msg: ~"Expecting ':' but got 'a'" })]);
    }

    #[test]
    fn parse_empty_list() {
        assert_stream_eq("le", [ListStart, ListEnd]);
    }

    #[test]
    fn parses_list_with_number() {
        assert_stream_eq("li2006ee", [ListStart, NumberValue(2006), ListEnd]);
    }

    #[test]
    fn parses_list_with_bytestring() {
        assert_stream_eq("l9:foobarbaze", [ListStart, ByteStringValue(bytes("foobarbaz")), ListEnd]);
    }

    #[test]
    fn parses_list_with_mixed_elements() {
        assert_stream_eq("l4:rusti2006ee",
                         [ListStart,
                          ByteStringValue(bytes("rust")),
                          NumberValue(2006),
                          ListEnd]);
    }

    #[test]
    fn parses_nested_lists() {
        assert_stream_eq("li1983el3:c++e3:oope",
                         [ListStart,
                          NumberValue(1983),
                          ListStart,
                          ByteStringValue(bytes("c++")),
                          ListEnd,
                          ByteStringValue(bytes("oop")),
                          ListEnd]);
    }

    #[test]
    fn parses_empty_dict() {
        assert_stream_eq("de", [DictStart, DictEnd]);
    }

    #[test]
    fn parses_dict_with_number_value() {
        assert_stream_eq("d3:fooi2006ee",
                         [DictStart,
                          DictKey(bytes("foo")),
                          NumberValue(2006),
                          DictEnd])
    }

    #[test]
    fn parses_dict_with_bytestring_value() {
        assert_stream_eq("d3:foo4:2006e",
                         [DictStart,
                          DictKey(bytes("foo")),
                          ByteStringValue(bytes("2006")),
                          DictEnd])
    }

    #[test]
    fn parses_dict_with_list_value() {
        assert_stream_eq("d3:fooli2006eee",
                         [DictStart,
                          DictKey(bytes("foo")),
                          ListStart,
                          NumberValue(2006),
                          ListEnd,
                          DictEnd]);
        assert_stream_eq("d1:ai123e1:bl3:foo4:baaree",
                        [DictStart,
                         DictKey(bytes("a")),
                         NumberValue(123),
                         DictKey(bytes("b")),
                         ListStart,
                         ByteStringValue(bytes("foo")),
                         ByteStringValue(bytes("baar")),
                         ListEnd,
                         DictEnd])
    }

    #[test]
    fn parses_nested_dicts() {
        assert_stream_eq("d3:food3:bari2006eee",
                         [DictStart,
                          DictKey(bytes("foo")),
                          DictStart,
                          DictKey(bytes("bar")),
                          NumberValue(2006),
                          DictEnd,
                          DictEnd])
    }

    #[test]
    fn parse_error_on_key_of_wrong_type() {
        assert_stream_eq("di2006ei1ee",
                         [DictStart,
                          ParseError(Error{ pos: 1, msg: ~"Expecting '0-9 or e' but got 'i'" })]);
        assert_stream_eq("dleei1ee",
                         [DictStart,
                          ParseError(Error{ pos: 1, msg: ~"Expecting '0-9 or e' but got 'l'" })]);
        assert_stream_eq("ddei1ee",
                         [DictStart,
                          ParseError(Error{ pos: 1, msg: ~"Expecting '0-9 or e' but got 'd'" })]);
    }

    #[test]
    fn parse_error_on_unmatched_value_ending() {
        let msg = ~"Unmatched value ending";
        assert_stream_eq("e", [ParseError(Error{ pos: 1, msg: msg.clone() })]);
        assert_stream_eq("lee", [ListStart, ListEnd, ParseError(Error{ pos: 3, msg: msg.clone() })]);
    }

    fn assert_decoded_eq(events: &[BencodeEvent], expected: Result<Bencode, Error>) {
        let mut parser = Parser::new(events.to_owned().move_iter());
        let result = parser.parse();
        assert_eq!(expected, result);
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
        assert_decoded_eq([ListStart, ListEnd], Ok(List(~[])));
    }

    #[test]
    fn decodes_list_with_elements() {
        assert_decoded_eq([ListStart,
                           NumberValue(1),
                           ListEnd], Ok(List(~[Number(1)])));
        assert_decoded_eq([ListStart,
                           ByteStringValue(bytes("str")),
                           NumberValue(11),
                           ListEnd], Ok(List(~[ByteString(bytes("str")),
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
                           Ok(List(~[List(~[Number(13)]),
                                     ByteString(bytes("rust"))])));
    }

    #[test]
    fn decodes_empty_dict() {
        assert_decoded_eq([DictStart, DictEnd], Ok(Dict(TreeMap::new())));
    }

    #[test]
    fn decodes_dict_with_value() {
        let mut map = TreeMap::new();
        map.insert(Key(bytes("foo")), ByteString(bytes("rust")));
        assert_decoded_eq([DictStart,
                           DictKey(bytes("foo")),
                           ByteStringValue(bytes("rust")),
                           DictEnd], Ok(Dict(map)));
    }

    #[test]
    fn decodes_dict_with_values() {
        let mut map = TreeMap::new();
        map.insert(Key(bytes("num")), Number(9));
        map.insert(Key(bytes("str")), ByteString(bytes("abc")));
        map.insert(Key(bytes("list")), List(~[Number(99)]));
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
        inner.insert(Key(bytes("inner")), Number(2));
        let mut outer = TreeMap::new();
        outer.insert(Key(bytes("dict")), Dict(inner));
        outer.insert(Key(bytes("outer")), Number(1));
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
        let err = Error{ pos: 1, msg: ~"error msg" };
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

    use self::test::BenchHarness;

    use std::io;
    use std::vec;

    use serialize::{Encodable, Decodable};

    use super::{Encoder, Decoder, Parser, StreamingParser};

    #[bench]
    fn encode_large_vec_of_uint(bh: &mut BenchHarness) {
        let v = vec::from_fn(100, |n| n);
        bh.iter(|| {
            let mut w = io::MemWriter::with_capacity(v.len() * 10);
            {
                let mut enc = Encoder::new(&mut w);
                v.encode(&mut enc);
            }
            w.unwrap()
        });
        bh.bytes = v.len() as u64 * 4;
    }


    #[bench]
    fn decode_large_vec_of_uint(bh: &mut BenchHarness) {
        let v = vec::from_fn(100, |n| n);
        let b = Encoder::buffer_encode(&v).unwrap();
        bh.iter(|| {
            let streaming_parser = StreamingParser::new(b.clone().move_iter());
            let mut parser = Parser::new(streaming_parser);
            let bencode = parser.parse().unwrap();
            let mut decoder = Decoder::new(&bencode);
            let result: ~[uint] = Decodable::decode(&mut decoder);
            result
        });
        bh.bytes = b.len() as u64 * 4;
    }
}

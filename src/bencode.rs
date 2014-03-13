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
#[feature(macro_rules)];

extern crate collections;
extern crate serialize;

use std::io;
use std::fmt;
use std::str;
use std::str::raw;
use std::vec_ng::Vec;

use serialize::{Encodable};

use collections::treemap::TreeMap;

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

#[deriving(Eq, Clone, TotalOrd, TotalEq, Ord, Show)]
pub struct Key(~[u8]);

pub type List = ~[Bencode];
pub type Dict = TreeMap<Key, Bencode>;

impl Bencode {
    pub fn to_writer(&self, writer: &mut io::Writer) -> io::IoResult<()> {
        let mut encoder = Encoder::new(writer);
        self.encode(&mut encoder);
        encoder.error
    }

    pub fn to_bytes(&self) -> ~[u8] {
        let mut writer = io::MemWriter::new();
        self.to_writer(&mut writer).unwrap();
        writer.unwrap()
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

pub fn from_buffer(buf: &[u8]) -> Result<Bencode, Error> {
    from_owned(buf.to_owned())
}

pub fn from_owned(buf: ~[u8]) -> Result<Bencode, Error> {
    let mut reader = io::MemReader::new(buf);
    from_iter(reader.bytes())
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
    priv depth: uint,
    priv current_writer: io::MemWriter,
    priv error: io::IoResult<()>,
    priv dict_stack: Vec<TreeMap<Key, ~[u8]>>,
}

impl<'a> Encoder<'a> {
    pub fn new(writer: &'a mut io::Writer) -> Encoder<'a> {
        Encoder {
            writer: writer,
            depth: 0,
            current_writer: io::MemWriter::new(),
            error: Ok(()),
            dict_stack: Vec::new()
        }
    }

    pub fn buffer_encode<T: Encodable<Encoder<'a>>>(val: &T) -> ~[u8] {
        let mut writer = io::MemWriter::new();
        {
            let mut encoder = Encoder::new(&mut writer);
            val.encode(&mut encoder);
        }
        writer.unwrap()
    }

    fn get_writer(&'a mut self) -> &'a mut io::Writer {
        if self.depth == 0 {
            &mut self.writer as &'a mut io::Writer
        } else {
            &mut self.current_writer as &'a mut io::Writer
        }
    }

    fn encode_dict(&mut self, dict: &TreeMap<Key, ~[u8]>) {
        for (key, value) in dict.iter() {
            key.encode(self);
            tryenc!(self.get_writer().write(*value));
        }
    }
}

impl<'a> serialize::Encoder for Encoder<'a> {
    fn emit_nil(&mut self) { tryenc!(write!(self.get_writer(), "0:")) }

    fn emit_uint(&mut self, v: uint) { self.emit_i64(v as i64); }

    fn emit_u8(&mut self, v: u8) { self.emit_i64(v as i64); }

    fn emit_u16(&mut self, v: u16) { self.emit_i64(v as i64); }

    fn emit_u32(&mut self, v: u32) { self.emit_i64(v as i64); }

    fn emit_u64(&mut self, v: u64) { self.emit_i64(v as i64); }

    fn emit_int(&mut self, v: int) { self.emit_i64(v as i64); }

    fn emit_i8(&mut self, v: i8) { self.emit_i64(v as i64); }

    fn emit_i16(&mut self, v: i16) { self.emit_i64(v as i64); }

    fn emit_i32(&mut self, v: i32) { self.emit_i64(v as i64); }

    fn emit_i64(&mut self, v: i64) { tryenc!(write!(self.get_writer(), "i{}e", v)) }

    fn emit_bool(&mut self, v: bool) {
        if v {
            self.emit_str("true");
        } else {
            self.emit_str("false");
        }
    }

    fn emit_f32(&mut self, v: f32) {
        self.emit_str(std::f32::to_str_hex(v));
    }

    fn emit_f64(&mut self, v: f64) {
        self.emit_str(std::f64::to_str_hex(v));
    }

    fn emit_char(&mut self, v: char) { self.emit_str(str::from_char(v)); }

    fn emit_str(&mut self, v: &str) {
        tryenc!(write!(self.get_writer(), "{}:", v.len()));
        tryenc!(self.get_writer().write(v.as_bytes()));
    }

    fn emit_enum(&mut self, _name: &str, _f: |&mut Encoder<'a>|) { unimplemented!(); }
    fn emit_enum_variant(&mut self, _v_name: &str, _v_id: uint, _len: uint, _f: |&mut Encoder<'a>|) { unimplemented!(); }
    fn emit_enum_variant_arg(&mut self, _a_idx: uint, _f: |&mut Encoder<'a>|) { unimplemented!(); }
    fn emit_enum_struct_variant(&mut self, _v_name: &str, _v_id: uint, _len: uint, _f: |&mut Encoder<'a>|) { unimplemented!(); }
    fn emit_enum_struct_variant_field(&mut self, _f_name: &str, _f_idx: uint, _f: |&mut Encoder<'a>|) { unimplemented!(); }

    fn emit_struct(&mut self, _name: &str, _len: uint, f: |&mut Encoder<'a>|) {
        tryenc!(write!(self.get_writer(), "d"));
        self.depth += 1;
        self.dict_stack.push(TreeMap::new());
        f(self);
        self.depth -= 1;
        let dict = self.dict_stack.pop().unwrap();
        self.encode_dict(&dict);
        tryenc!(write!(self.get_writer(), "e"));
    }

    fn emit_struct_field(&mut self, f_name: &str, _f_idx: uint, f: |&mut Encoder<'a>|) {
        f(self);
        let data = std::mem::replace(&mut self.current_writer, io::MemWriter::new());
        let dict = self.dict_stack.mut_last().unwrap();
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
        tryenc!(write!(self.get_writer(), "l"));
        f(self);
        tryenc!(write!(self.get_writer(), "e"));
    }

    fn emit_seq_elt(&mut self, _idx: uint, f: |this: &mut Encoder<'a>|) {
        f(self);
    }

    fn emit_map(&mut self, _len: uint, f: |&mut Encoder<'a>|) {
        tryenc!(write!(self.get_writer(), "d"));
        f(self);
        tryenc!(write!(self.get_writer(), "e"));
    }

    fn emit_map_elt_key(&mut self, _idx: uint, f: |&mut Encoder<'a>|) {
        f(self);
    }

    fn emit_map_elt_val(&mut self, _idx: uint, f: |&mut Encoder<'a>|) {
        f(self);
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
enum ExpectedToken {
    ValueToken,
    KeyToken
}

pub struct StreamingParser<T> {
    priv reader: T,
    priv pos: u32,
    priv decoded: u32,
    priv end: bool,
    priv curr: Option<u8>,
    priv nesting: Vec<BencodeEvent>,
    priv expected: ExpectedToken
}

macro_rules! expect(($ch:pat, $ex:expr) => (
    match self.curr_char() {
        Some($ch) => self.next_byte(),
        _ => return self.error($ex)
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
            nesting: Vec::new(),
            expected: ValueToken
        }
    }

    fn next_byte(&mut self) {
        self.pos += 1;
        self.curr = self.reader.next();
        self.end = self.curr.is_none();
    }
    
    fn next_bytes(&mut self, len: uint) -> Result<~[u8], Error> {
        let mut bytes = ~[];
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

    fn parse_digit(&self, ch: char, num: &mut i64) {
        *num *= 10;
        *num += ch.to_digit(10).unwrap() as i64;
    }

    fn parse_bytestring(&mut self) -> Result<~[u8], Error> {
        let len = try!(self.parse_number_unsigned());
        expect!(':', ~":");
        let bytes = try!(self.next_bytes(len as uint));
        Ok(bytes)
    }
}

macro_rules! try_result(($e:expr) => (
    match $e {
        Ok(e) => Some(e),
        Err(err) => {
            self.end = true;
            return Some(ParseError(err))
        }
    }
))

macro_rules! check_nesting(() => (
    if self.decoded > 0 && self.nesting.len() == 0 {
        return try_result!(self.error_msg(~"Only one value allowed outside of containers"))
    }
))

impl<T: Iterator<u8>> Iterator<BencodeEvent> for StreamingParser<T> {
    fn next(&mut self) -> Option<BencodeEvent> {
        if self.end {
            return None
        }
        match self.curr_char() {
            Some('i') => {
                check_nesting!();
                if self.expected == KeyToken {
                    return try_result!(self.error_msg(~"Wrong key type: integer"));
                }
                self.next_byte();
                self.decoded += 1;
                let res = try_result!(self.parse_number());
                res.map(|v| NumberValue(v))
            }
            Some('0' .. '9') => {
                check_nesting!();
                self.decoded += 1;
                let res = try_result!(self.parse_bytestring());
                match self.expected {
                    ValueToken => res.map(|v| ByteStringValue(v)),
                    KeyToken => {
                        self.expected = ValueToken;
                        res.map(|v| DictKey(v))
                    }
                }
            }
            Some('l') => {
                check_nesting!();
                if self.expected == KeyToken {
                    return try_result!(self.error_msg(~"Wrong key type: list"));
                }
                self.next_byte();
                self.decoded += 1;
                self.nesting.push(ListStart);
                Some(ListStart)
            }
            Some('d') => {
                check_nesting!();
                if self.expected == KeyToken {
                    return try_result!(self.error_msg(~"Wrong key type: dict"));
                }
                self.next_byte();
                self.decoded += 1;
                self.nesting.push(DictStart);
                self.expected = KeyToken;
                Some(DictStart)
            }
            Some('e') => {
                self.next_byte();
                match self.nesting.pop() {
                    Some(ListStart) => Some(ListEnd),
                    Some(DictStart) => Some(DictEnd),
                    _ => return try_result!(self.error_msg(~"Unmatched value ending"))
                }
            }
            _ => try_result!(self.error(~"i or 0-9 or l or d or e"))
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
            x => fail!("Unreachable but got {}", x)
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
                x => fail!("Unreachable but got {}", x)
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
                x => fail!("Unreachable but got {}", x)
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
    priv stack: Vec<&'a Bencode>,
}

impl<'a> Decoder<'a> {
    pub fn new(bencode: &'a Bencode) -> Decoder<'a> {
        Decoder {
            stack: Vec::from_slice([bencode])
        }
    }

    fn try_read<T: FromBencode>(&mut self) -> T {
        self.stack.pop().and_then(|b| FromBencode::from_bencode(b)).unwrap()
    }
}

impl<'a> serialize::Decoder for Decoder<'a> {
    fn read_nil(&mut self) { self.try_read() }

    fn read_uint(&mut self) -> uint { self.try_read() }

    fn read_u8(&mut self) -> u8 { self.try_read() }

    fn read_u16(&mut self) -> u16 { self.try_read() }

    fn read_u32(&mut self) -> u32 { self.try_read() }

    fn read_u64(&mut self) -> u64 { self.try_read() }

    fn read_int(&mut self) -> int { self.try_read() }

    fn read_i8(&mut self) -> i8 { self.try_read() }

    fn read_i16(&mut self) -> i16 { self.try_read() }

    fn read_i32(&mut self) -> i32 { self.try_read() }

    fn read_i64(&mut self) -> i64 { self.try_read() } 

    fn read_bool(&mut self) -> bool { self.try_read() }

    fn read_f32(&mut self) -> f32 { self.try_read() }

    fn read_f64(&mut self) -> f64 { self.try_read() }

    fn read_char(&mut self) -> char { self.try_read() }

    fn read_str(&mut self) -> ~str {
        match self.stack.pop() {
            Some(&ByteString(ref v)) => {
                match str::from_utf8_owned(v.clone()) {
                    Some(s) => s,
                    _ => fail!()
                }
            }
            _ => fail!()
        }
    }

    fn read_enum<T>(&mut self, _name: &str, _f: |&mut Decoder<'a>| -> T) -> T { unimplemented!(); }
    fn read_enum_variant<T>(&mut self, _names: &[&str], _f: |&mut Decoder<'a>, uint| -> T) -> T { unimplemented!(); }
    fn read_enum_variant_arg<T>(&mut self, _a_idx: uint, _f: |&mut Decoder<'a>| -> T) -> T { unimplemented!(); }
    fn read_enum_struct_variant<T>(&mut self, _names: &[&str], _f: |&mut Decoder<'a>, uint| -> T) -> T { unimplemented!(); }
    fn read_enum_struct_variant_field<T>(&mut self, _f_name: &str, _f_idx: uint, _f: |&mut Decoder<'a>| -> T) -> T { unimplemented!(); }
    fn read_struct<T>(&mut self, _s_name: &str, _len: uint, _f: |&mut Decoder<'a>| -> T) -> T { unimplemented!(); }
    fn read_struct_field<T>(&mut self, _f_name: &str, _f_idx: uint, _f: |&mut Decoder<'a>| -> T) -> T { unimplemented!(); }
    fn read_tuple<T>(&mut self, _f: |&mut Decoder<'a>, uint| -> T) -> T { unimplemented!(); }
    fn read_tuple_arg<T>(&mut self, _a_idx: uint, _f: |&mut Decoder<'a>| -> T) -> T { unimplemented!(); }
    fn read_tuple_struct<T>(&mut self, _s_name: &str, _f: |&mut Decoder<'a>, uint| -> T) -> T { unimplemented!(); }
    fn read_tuple_struct_arg<T>(&mut self, _a_idx: uint, _f: |&mut Decoder<'a>| -> T) -> T { unimplemented!(); }
    fn read_option<T>(&mut self, _f: |&mut Decoder<'a>, bool| -> T) -> T { unimplemented!(); }

    fn read_seq<T>(&mut self, f: |&mut Decoder<'a>, uint| -> T) -> T {
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

    fn read_seq_elt<T>(&mut self, _idx: uint, f: |&mut Decoder<'a>| -> T) -> T { f(self) }

    fn read_map<T>(&mut self, _f: |&mut Decoder<'a>, uint| -> T) -> T { unimplemented!(); }
    fn read_map_elt_key<T>(&mut self, _idx: uint, _f: |&mut Decoder<'a>| -> T) -> T { unimplemented!(); }
    fn read_map_elt_val<T>(&mut self, _idx: uint, _f: |&mut Decoder<'a>| -> T) -> T { unimplemented!(); }
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::str::raw;
    use serialize::{Encodable, Decodable};
    use collections::treemap::TreeMap;

    use super::{Bencode, ToBencode, Error};
    use super::{Encoder, ByteString, List, Number, Dict, Key};
    use super::{StreamingParser, BencodeEvent, NumberValue, ByteStringValue,
                ListStart, ListEnd, DictStart, DictKey, DictEnd, ParseError};
    use super::{Parser, Decoder};
    use super::alphanum_to_str;

    #[deriving(Eq, Show, Encodable)]
    struct SimpleStruct {
        a: uint,
        b: ~[~str],
    }

    macro_rules! assert_encoding(($value:expr, $expected:expr) => ({
        let v = $value;
        let result = Encoder::buffer_encode(&v);
        assert_eq!($expected, result);
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
                let value = $val;
                let bencode = value.to_bencode();
                let enc_bencode = bencode.to_bytes();
                let enc = Encoder::buffer_encode(&value);
                assert_eq!(enc, enc_bencode);
            };)+
        }
    })

    macro_rules! assert_identity(($value:expr) => ({
        let value = $value;
        let encoded = Encoder::buffer_encode(&value);
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

    #[test]
    fn encodes_number_zero() {
        assert_eq!(Number(0).to_bytes(), bytes("i0e"));
    }

    #[test]
    fn encodes_positive_numbers() {
        assert_eq!(Number(-5).to_bytes(), bytes("i-5e"));
        assert_eq!(Number(::std::i64::MIN).to_bytes(), format!("i{}e", ::std::i64::MIN).as_bytes().to_owned());
    }

    #[test]
    fn encodes_negative_numbers() {
        assert_eq!(Number(5).to_bytes(), bytes("i5e"));
        assert_eq!(Number(::std::i64::MAX).to_bytes(), format!("i{}e", ::std::i64::MAX).as_bytes().to_owned());
    }

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

    #[test]
    fn encodes_empty_bytestring() {
        assert_eq!(ByteString(~[]).to_bytes(), bytes("0:"));
    }

    #[test]
    fn encodes_nonempty_bytestring() {
        assert_eq!(ByteString((~"abc").into_bytes()).to_bytes(), bytes("3:abc"));
        assert_eq!(ByteString(~[0, 1, 2, 3]).to_bytes(), bytes("4:") + ~[0u8, 1, 2, 3]);
    }

    #[test]
    fn encodes_empty_list() {
        assert_eq!(List(~[]).to_bytes(), bytes("le"));
    }

    #[test]
    fn encodes_nonempty_list() {
        assert_eq!(List(~[Number(1)]).to_bytes(), bytes("li1ee"));
        assert_eq!(List(~[ByteString((~"foobar").into_bytes()),
                          Number(-1)]).to_bytes(), bytes("l6:foobari-1ee"));
    }

    #[test]
    fn encodes_nested_list() {
        assert_eq!(List(~[List(~[])]).to_bytes(), bytes("llee"));
        let list = List(~[Number(1988), List(~[Number(2014)])]);
        assert_eq!(list.to_bytes(), bytes("li1988eli2014eee"));
    }

    #[test]
    fn encodes_empty_dict() {
        assert_eq!(Dict(TreeMap::new()).to_bytes(), bytes("de"));
    }

    #[test]
    fn encodes_dict_with_items() {
        let mut m = TreeMap::new();
        m.insert(Key((~"k1").into_bytes()), Number(1));
        assert_eq!(Dict(m.clone()).to_bytes(), bytes("d2:k1i1ee"));
        m.insert(Key((~"k2").into_bytes()), ByteString(~[0, 0]));
        assert_eq!(Dict(m.clone()).to_bytes(), bytes("d2:k1i1e2:k22:\0\0e"));
    }

    #[test]
    fn encodes_nested_dict() {
        let mut outer = TreeMap::new();
        let mut inner = TreeMap::new();
        inner.insert(Key((~"val").into_bytes()), ByteString(~[68, 0, 90]));
        outer.insert(Key((~"inner").into_bytes()), Dict(inner));
        assert_eq!(Dict(outer).to_bytes(), bytes("d5:innerd3:val3:D\0Zee"));
    }

    #[test]
    fn encodes_dict_fields_in_sorted_order() {
        let mut m = TreeMap::new();
        m.insert(Key((~"z").into_bytes()), Number(1));
        m.insert(Key((~"abd").into_bytes()), Number(3));
        m.insert(Key((~"abc").into_bytes()), Number(2));
        assert_eq!(Dict(m).to_bytes(), bytes("d3:abci2e3:abdi3e1:zi1ee"));
    }

    #[test]
    fn encodes_encodable_struct() {
        let mut writer = io::MemWriter::new();
        {
            let mut encoder = Encoder::new(&mut writer);
            let s = SimpleStruct {
                b: ~[~"foo", ~"baar"],
                a: 123
            };
            s.encode(&mut encoder);
        }
        assert_eq!(writer.unwrap(), bytes("d1:ai123e1:bl3:foo4:baaree"));
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
        let mut writer = io::MemWriter::new();
        {
            let mut encoder = Encoder::new(&mut writer);
            let s = OrderedStruct {
                z: 4,
                a: 1,
                ab: 3,
                aa: 2
            };
            s.encode(&mut encoder);
        }
        assert_eq!(writer.unwrap(), bytes("d1:ai1e2:aai2e2:abi3e1:zi4ee"));
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
    fn parse_error_on_key_of_of_wrong_type() {
        assert_stream_eq("di2006ei1ee",
                         [DictStart,
                          ParseError(Error{ pos: 1, msg: ~"Wrong key type: integer" })]);
        assert_stream_eq("dleei1ee",
                         [DictStart,
                          ParseError(Error{ pos: 1, msg: ~"Wrong key type: list" })]);
        assert_stream_eq("ddei1ee",
                         [DictStart,
                          ParseError(Error{ pos: 1, msg: ~"Wrong key type: dict" })]);
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

    #[test]
    fn identity_encode_decode_unit() {
        assert_identity!(());
    }

    #[test]
    fn identity_encode_decode_int() {
        assert_identity!(99i);
        assert_identity!(::std::int::MIN);
        assert_identity!(::std::int::MAX);
    }

    #[test]
    fn identity_encode_decode_i8() {
        assert_identity!(99i8);
        assert_identity!(::std::i8::MIN);
        assert_identity!(::std::i8::MAX);
    }

    #[test]
    fn identity_encode_decode_i16() {
        assert_identity!(99i16);
        assert_identity!(::std::i16::MIN);
        assert_identity!(::std::i16::MAX);
    }

    #[test]
    fn identity_encode_decode_i32() {
        assert_identity!(99i32);
        assert_identity!(::std::i32::MIN);
        assert_identity!(::std::i32::MAX);
    }

    #[test]
    fn identity_encode_decode_i64() {
        assert_identity!(99i64);
        assert_identity!(::std::i64::MIN);
        assert_identity!(::std::i64::MAX);
    }

    #[test]
    fn identity_encode_decode_uint() {
        assert_identity!(99u);
        assert_identity!(::std::uint::MIN);
        assert_identity!(::std::uint::MAX);
    }

    #[test]
    fn identity_encode_decode_u8() {
        assert_identity!(99u8);
        assert_identity!(::std::u8::MIN);
        assert_identity!(::std::u8::MAX);
    }

    #[test]
    fn identity_encode_decode_u16() {
        assert_identity!(99u16);
        assert_identity!(::std::u16::MIN);
        assert_identity!(::std::u16::MAX);
    }

    #[test]
    fn identity_encode_decode_u32() {
        assert_identity!(99u32);
        assert_identity!(::std::u32::MIN);
        assert_identity!(::std::u32::MAX);
    }

    #[test]
    fn identity_encode_decode_u64() {
        assert_identity!(99u64);
        assert_identity!(::std::u64::MIN);
        assert_identity!(::std::u64::MAX);
    }

    #[test]
    fn identity_encode_decode_f32() {
        assert_identity!(99.0f32);
        assert_identity!(99.13f32);
        assert_identity!(::std::f32::MIN_VALUE as f32);
        assert_identity!(::std::f32::MAX_VALUE as f32);
        assert_identity!(-::std::f32::MIN_VALUE as f32);
        assert_identity!(-::std::f32::MAX_VALUE as f32);
    }

    #[test]
    fn identity_encode_decode_f64() {
        assert_identity!(99.0f64);
        assert_identity!(99.13f64);
        assert_identity!(::std::f64::MIN_VALUE);
        assert_identity!(::std::f64::MAX_VALUE);
        assert_identity!(-::std::f64::MIN_VALUE);
        assert_identity!(-::std::f64::MAX_VALUE);
    }

    #[test]
    fn identity_encode_decode_bool() {
        assert_identity!(true);
        assert_identity!(false);
    }

    #[test]
    fn identity_encode_decode_str() {
        assert_identity!(~"");
        assert_identity!(~"abc");
        assert_identity!(~"Löwe 老虎 Léopard");
    }

    #[test]
    fn identity_encode_decode_owned_vec() {
        let empty: ~[int] = ~[];
        assert_identity!(empty);
        assert_identity!(~[~"a", ~"ab", ~"abc"]);
        assert_identity!(~[1.1, 1.2, 1.3]);
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
        let b = Encoder::buffer_encode(&v);
        bh.iter(|| {
            let mut reader = io::MemReader::new(b.clone());
            let streaming_parser = StreamingParser::new(reader.bytes());
            let mut parser = Parser::new(streaming_parser);
            let bencode = parser.parse().unwrap();
            let mut decoder = Decoder::new(&bencode);
            let result: ~[uint] = Decodable::decode(&mut decoder);
            result
        });
        bh.bytes = b.len() as u64 * 4;
    }
}

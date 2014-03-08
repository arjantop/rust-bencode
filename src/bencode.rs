#[crate_id = "bencode"];
#[crate_type = "rlib"];
#[crate_type = "dylib"];
#[deny(warnings)];
#[feature(macro_rules)];
extern crate collections;
extern crate serialize;

use std::io;
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
    Dict(Dict)
}

#[deriving(Eq, Clone, TotalOrd, TotalEq)]
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
    fn from_bencode(bencode: &Bencode) -> Option<Self>;
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

impl ToBencode for ~str {
    fn to_bencode(&self) -> Bencode { ByteString((*self).as_bytes().into_owned()) }
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

macro_rules! try(($e:expr) => (
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
    priv dict_stack: Vec<TreeMap<Key, ~[u8]>>
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
            try!(self.get_writer().write(*value));
        }
    }
}

impl<'a> serialize::Encoder for Encoder<'a> {
    fn emit_nil(&mut self) { unimplemented!(); }

    fn emit_uint(&mut self, v: uint) { self.emit_i64(v as i64); }

    fn emit_u8(&mut self, v: u8) { self.emit_i64(v as i64); }

    fn emit_u16(&mut self, v: u16) { self.emit_i64(v as i64); }

    fn emit_u32(&mut self, v: u32) { self.emit_i64(v as i64); }

    fn emit_u64(&mut self, v: u64) { self.emit_i64(v as i64); }

    fn emit_int(&mut self, v: int) { self.emit_i64(v as i64); }

    fn emit_i8(&mut self, v: i8) { self.emit_i64(v as i64); }

    fn emit_i16(&mut self, v: i16) { self.emit_i64(v as i64); }

    fn emit_i32(&mut self, v: i32) { self.emit_i64(v as i64); }

    fn emit_i64(&mut self, v: i64) { try!(write!(self.get_writer(), "i{}e", v)) }

    fn emit_bool(&mut self, v: bool) {
        if v {
            try!(write!(self.get_writer(), "true"));
        } else {
            try!(write!(self.get_writer(), "false"));
        }
    }

    fn emit_f64(&mut self, _v: f64) { unimplemented!(); }
    fn emit_f32(&mut self, _v: f32) { unimplemented!(); }

    fn emit_char(&mut self, v: char) { self.emit_str(str::from_char(v)); }

    fn emit_str(&mut self, v: &str) {
        try!(write!(self.get_writer(), "{}:", v.len()));
        try!(self.get_writer().write(v.as_bytes()));
    }

    fn emit_enum(&mut self, _name: &str, _f: |&mut Encoder<'a>|) { unimplemented!(); }
    fn emit_enum_variant(&mut self, _v_name: &str, _v_id: uint, _len: uint, _f: |&mut Encoder<'a>|) { unimplemented!(); }
    fn emit_enum_variant_arg(&mut self, _a_idx: uint, _f: |&mut Encoder<'a>|) { unimplemented!(); }
    fn emit_enum_struct_variant(&mut self, _v_name: &str, _v_id: uint, _len: uint, _f: |&mut Encoder<'a>|) { unimplemented!(); }
    fn emit_enum_struct_variant_field(&mut self, _f_name: &str, _f_idx: uint, _f: |&mut Encoder<'a>|) { unimplemented!(); }

    fn emit_struct(&mut self, _name: &str, _len: uint, f: |&mut Encoder<'a>|) {
        try!(write!(self.get_writer(), "d"));
        self.depth += 1;
        self.dict_stack.push(TreeMap::new());
        f(self);
        self.depth -= 1;
        let dict = self.dict_stack.pop().unwrap();
        self.encode_dict(&dict);
        try!(write!(self.get_writer(), "e"));
    }

    fn emit_struct_field(&mut self, f_name: &str, _f_idx: uint, f: |&mut Encoder<'a>|) {
        f(self);
        {
            let data = self.current_writer.get_ref();
            let dict = self.dict_stack.mut_last().unwrap();
            dict.insert(Key(f_name.as_bytes().to_owned()), data.to_owned());
        }
        self.current_writer = io::MemWriter::new();
    }

    fn emit_tuple(&mut self, _len: uint, _f: |&mut Encoder<'a>|) { unimplemented!(); }
    fn emit_tuple_arg(&mut self, _idx: uint, _f: |&mut Encoder<'a>|) { unimplemented!(); }
    fn emit_tuple_struct(&mut self, _name: &str, _len: uint, _f: |&mut Encoder<'a>|) { unimplemented!(); }
    fn emit_tuple_struct_arg(&mut self, _f_idx: uint, _f: |&mut Encoder<'a>|) { unimplemented!(); }
    fn emit_option(&mut self, _f: |&mut Encoder<'a>|) { unimplemented!(); }
    fn emit_option_none(&mut self) { unimplemented!(); }
    fn emit_option_some(&mut self, _f: |&mut Encoder<'a>|) { unimplemented!(); }

    fn emit_seq(&mut self, _len: uint, f: |this: &mut Encoder<'a>|) {
        try!(write!(self.get_writer(), "l"));
        f(self);
        try!(write!(self.get_writer(), "e"));
    }

    fn emit_seq_elt(&mut self, _idx: uint, f: |this: &mut Encoder<'a>|) {
        f(self);
    }

    fn emit_map(&mut self, _len: uint, f: |&mut Encoder<'a>|) {
        try!(write!(self.get_writer(), "d"));
        f(self);
        try!(write!(self.get_writer(), "e"));
    }

    fn emit_map_elt_key(&mut self, _idx: uint, f: |&mut Encoder<'a>|) {
        f(self);
    }

    fn emit_map_elt_val(&mut self, _idx: uint, f: |&mut Encoder<'a>|) {
        f(self);
    }
}

#[cfg(test)]
mod test {
    use std::io;
    use serialize::{Encodable};
    use collections::treemap::TreeMap;

    use super::{Encoder, ByteString, List, Number, Dict, Key};

    #[deriving(Eq, Show, Encodable)]
    struct SimpleStruct {
        a: uint,
        b: ~[~str]
    }

    #[test]
    fn can_encode_number_zero() {
        assert_eq!(Number(0).to_bytes(), bytes!("i0e").to_owned());
    }

    #[test]
    fn can_encode_positive_numbers() {
        assert_eq!(Number(-5).to_bytes(), bytes!("i-5e").to_owned());
        assert_eq!(Number(::std::i64::MIN).to_bytes(), format!("i{}e", ::std::i64::MIN).as_bytes().to_owned());
    }

    #[test]
    fn can_encode_negative_numbers() {
        assert_eq!(Number(5).to_bytes(), bytes!("i5e").to_owned());
        assert_eq!(Number(::std::i64::MAX).to_bytes(), format!("i{}e", ::std::i64::MAX).as_bytes().to_owned());
    }

    #[test]
    fn can_encode_empty_bytestring() {
        assert_eq!(ByteString(~[]).to_bytes(), bytes!("0:").to_owned());
    }

    #[test]
    fn can_encode_nonempty_bytestring() {
        assert_eq!(ByteString((~"abc").into_bytes()).to_bytes(), bytes!("3:abc").to_owned());
        assert_eq!(ByteString(~[0, 1, 2, 3]).to_bytes(), bytes!("4:") + ~[0u8, 1, 2, 3]);
    }

    #[test]
    fn can_encode_empty_list() {
        assert_eq!(List(~[]).to_bytes(), bytes!("le").to_owned());
    }

    #[test]
    fn can_encode_nonempty_list() {
        assert_eq!(List(~[Number(1)]).to_bytes(), bytes!("li1ee").to_owned());
        assert_eq!(List(~[ByteString((~"foobar").into_bytes()),
                          Number(-1)]).to_bytes(), bytes!("l6:foobari-1ee").to_owned());
    }

    #[test]
    fn can_encode_nested_list() {
        assert_eq!(List(~[List(~[])]).to_bytes(), bytes!("llee").to_owned());
        let list = List(~[Number(1988), List(~[Number(2014)])]);
        assert_eq!(list.to_bytes(), bytes!("li1988eli2014eee").to_owned());
    }

    #[test]
    fn can_encode_empty_dict() {
        assert_eq!(Dict(TreeMap::new()).to_bytes(), bytes!("de").to_owned());
    }

    #[test]
    fn can_encode_dict_with_items() {
        let mut m = TreeMap::new();
        m.insert(Key((~"k1").into_bytes()), Number(1));
        assert_eq!(Dict(m.clone()).to_bytes(), bytes!("d2:k1i1ee").to_owned());
        m.insert(Key((~"k2").into_bytes()), ByteString(~[0, 0]));
        assert_eq!(Dict(m.clone()).to_bytes(), bytes!("d2:k1i1e2:k22:\0\0e").to_owned());
    }

    #[test]
    fn can_encode_nested_dict() {
        let mut outer = TreeMap::new();
        let mut inner = TreeMap::new();
        inner.insert(Key((~"val").into_bytes()), ByteString(~[68, 0, 90]));
        outer.insert(Key((~"inner").into_bytes()), Dict(inner));
        assert_eq!(Dict(outer).to_bytes(), bytes!("d5:innerd3:val3:D\0Zee").to_owned());
    }

    #[test]
    fn should_encode_dict_fields_in_sorted_order() {
        let mut m = TreeMap::new();
        m.insert(Key((~"z").into_bytes()), Number(1));
        m.insert(Key((~"abd").into_bytes()), Number(3));
        m.insert(Key((~"abc").into_bytes()), Number(2));
        assert_eq!(Dict(m).to_bytes(), bytes!("d3:abci2e3:abdi3e1:zi1ee").to_owned());
    }

    #[test]
    fn can_encode_encodable_struct() {
        let mut writer = io::MemWriter::new();
        {
            let mut encoder = Encoder::new(&mut writer);
            let s = SimpleStruct {
                b: ~[~"foo", ~"baar"],
                a: 123
            };
            s.encode(&mut encoder);
        }
        assert_eq!(writer.unwrap(), bytes!("d1:ai123e1:bl3:foo4:baaree").to_owned());
    }

    #[test]
    fn should_encode_struct_fields_in_sorted_order() {
        #[deriving(Encodable)]
        struct OrderedStruct {
            z: int,
            a: int,
            ab: int,
            aa: int
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
        assert_eq!(writer.unwrap(), bytes!("d1:ai1e2:aai2e2:abi3e1:zi4ee").to_owned());
    }
}

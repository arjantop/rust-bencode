use std::str::raw;
use std::str;
use std::fmt;

use serialize;
use serialize::{Encodable, Decodable, Decoder};

use super::{DecoderError, StringEncoding};

#[deriving(Eq, PartialEq, Clone, Ord, PartialOrd, Hash)]
pub struct ByteString(Vec<u8>);

impl ByteString {
    pub fn from_str(s: &str) -> ByteString {
        ByteString(s.as_bytes().to_vec())
    }

    pub fn from_slice(s: &[u8]) -> ByteString {
        ByteString(s.to_vec())
    }

    pub fn from_vec(s: Vec<u8>) -> ByteString {
        ByteString(s)
    }

    pub fn as_slice<'a>(&'a self) -> &'a [u8] {
        self.as_slice()
    }

    pub fn unwrap(self) -> Vec<u8> {
        let ByteString(v) = self;
        v
    }
}

impl fmt::Show for ByteString {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &ByteString(ref v) => match str::from_utf8(v.as_slice()) {
                Some(s) => write!(fmt, "ByteString(\"{}\")", s),
                None    => write!(fmt, "ByteString({})", v),
            }
        }
    }
}

impl<E, S: serialize::Encoder<E>> Encodable<S, E> for ByteString {
    fn encode(&self, e: &mut S) -> Result<(), E> {
        let &ByteString(ref key) = self;
        e.emit_str(unsafe { raw::from_utf8(key.as_slice()) })
    }
}

impl<'a> Decodable<super::Decoder<'a>, DecoderError> for ByteString {
    fn decode(d: &mut super::Decoder<'a>) -> Result<ByteString, DecoderError> {
        d.read_str().map(|s| ByteString(s.into_bytes())).or_else(|e| {
            match e {
                StringEncoding(v) => Ok(ByteString(v)),
                err => Err(err)
            }
        })
    }
}

#[cfg(test)]
mod test {
    use serialize::Decodable;

    use super::ByteString;
    use super::super::{Encoder, from_vec, Decoder};

    #[test]
    fn test_encoding_bytestring_with_invalid_utf8() {
        let bs = ByteString::from_slice([45, 18, 128]);
        let encoded = match Encoder::buffer_encode(&bs) {
            Ok(e) => e,
            Err(err) => panic!("Unexpected failure: {}", err)
        };
        assert_eq!(vec![51, 58, 45, 18, 128], encoded);
    }

    #[test]
    fn test_decoding_bytestring_with_invalid_utf8() {
        let encoded = vec![51, 58, 45, 18, 128];
        let bencode = from_vec(encoded).unwrap();
        let mut decoder = Decoder::new(&bencode);
        let result: Result<ByteString, _> = Decodable::decode(&mut decoder);
        assert_eq!(Ok(ByteString::from_slice([45, 18, 128])), result);
    }
}

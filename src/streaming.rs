#[deriving(Show, Eq, Clone)]
pub enum BencodeEvent {
    NumberValue(i64),
    ByteStringValue(Vec<u8>),
    ListStart,
    ListEnd,
    DictStart,
    DictKey(Vec<u8>),
    DictEnd,
    ParseError(Error),
}

#[deriving(Show, Eq, Clone)]
pub struct Error {
    pub pos: u32,
    pub msg: ~str,
}

#[deriving(Show, Eq, Clone)]
pub enum BencodePosition {
    ListPosition,
    KeyPosition,
    ValuePosition
}

pub struct StreamingParser<T> {
    reader: T,
    pos: u32,
    decoded: u32,
    end: bool,
    curr: Option<u8>,
    stack: Vec<BencodePosition>,
}

macro_rules! expect(($ch:pat, $ex:expr) => (
    match self.curr_char() {
        Some($ch) => self.next_byte(),
        _ => return self.error($ex)
    }
))

macro_rules! check_nesting(() => (
    if self.decoded > 0 && self.stack.len() == 0 {
        return self.error_msg("Only one value allowed outside of containers".to_owned())
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

    fn next_bytes(&mut self, len: uint) -> Result<Vec<u8>, Error> {
        let mut bytes = Vec::with_capacity(len);
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
            None => "EOF".to_owned()
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
        expect!('e', "e".to_owned());
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
            _ => return self.error("0-9".to_owned())
        };
        Ok(num)
    }

    #[inline(always)]
    fn parse_digit(&self, ch: char, num: &mut i64) {
        *num *= 10;
        *num += ch.to_digit(10).unwrap() as i64;
    }

    #[inline(always)]
    fn parse_bytestring(&mut self) -> Result<Vec<u8>, Error> {
        let len = try!(self.parse_number_unsigned());
        expect!(':', ":".to_owned());
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
            _ => return self.error_msg("Unmatched value ending".to_owned())
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
            _ => self.error("0-9 or e".to_owned())
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
            _ => self.error("i or 0-9 or l or d or e".to_owned())
        }
    }

    fn empty_input(&self) -> bool {
        self.curr == None && self.decoded == 0
    }
}

impl<T: Iterator<u8>> Iterator<BencodeEvent> for StreamingParser<T> {
    fn next(&mut self) -> Option<BencodeEvent> {
        if self.end || self.empty_input() {
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

#[cfg(test)]
mod test {
    use std::str::raw;

    use super::{StreamingParser, Error};
    use super::{BencodeEvent, NumberValue, ByteStringValue, ListStart,
                    ListEnd, DictStart, DictKey, DictEnd, ParseError};
    use super::alphanum_to_str;

    fn bytes(s: &str) -> Vec<u8> {
        Vec::from_slice(s.as_bytes())
    }

    fn assert_stream_eq(encoded: &str, expected: &[BencodeEvent]) {
        let mut parser = StreamingParser::new(encoded.bytes());
        let result = parser.collect::<Vec<_>>();
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
    fn parses_empty_input() {
        assert_stream_eq("", []);
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
        assert_stream_eq("i10", [ParseError(Error{ pos: 3, msg: "Expecting 'e' but got 'EOF'".to_owned() })]);
    }

    #[test]
    fn parse_error_on_number_with_leading_zero() {
        assert_stream_eq("i0215e", [ParseError(Error{ pos: 2, msg: "Expecting 'e' but got '2'".to_owned() })]);
    }

    #[test]
    fn parse_error_on_empty_number() {
        assert_stream_eq("ie", [ParseError(Error{ pos: 1, msg: "Expecting '0-9' but got 'e'".to_owned() })]);
    }

    #[test]
    fn parse_error_on_more_than_one_value_outside_of_containers() {
        let msg = "Only one value allowed outside of containers".to_owned();
        assert_stream_eq("i1ei2e", [NumberValue(1), ParseError(Error{ pos: 3, msg: msg.clone() })]);
        assert_stream_eq("i10eli2ee", [NumberValue(10), ParseError(Error{ pos: 4, msg: msg.clone() })]);
        assert_stream_eq("1:ade", [ByteStringValue(bytes("a")), ParseError(Error{ pos: 3, msg: msg.clone() })]);
        assert_stream_eq("3:foo3:bar", [ByteStringValue(bytes("foo")),
                                        ParseError(Error{ pos: 5, msg: msg.clone() })]);
    }

    #[test]
    fn parses_empty_bytestring() {
        assert_stream_eq("0:", [ByteStringValue(Vec::new())]);
    }

    #[test]
    fn parses_short_bytestring() {
        assert_stream_eq("6:abcdef", [ByteStringValue(bytes("abcdef"))]);
    }

    #[test]
    fn parses_long_bytestring() {
        let long = "baz".repeat(10);
        assert_stream_eq(format!("{}:{}", long.len(), long), [ByteStringValue(bytes(long))]);
    }

    #[test]
    fn parse_error_on_too_short_data() {
        assert_stream_eq("5:abcd", [ParseError(Error { pos: 6, msg: "Expecting 5 bytes but only got 4".to_owned() })]);
    }

    #[test]
    fn parse_error_on_malformed_bytestring() {
        assert_stream_eq("3abc", [ParseError(Error { pos: 1, msg: "Expecting ':' but got 'a'".to_owned() })]);
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
                          ParseError(Error{ pos: 1, msg: "Expecting '0-9 or e' but got 'i'".to_owned() })]);
        assert_stream_eq("dleei1ee",
                         [DictStart,
                          ParseError(Error{ pos: 1, msg: "Expecting '0-9 or e' but got 'l'".to_owned() })]);
        assert_stream_eq("ddei1ee",
                         [DictStart,
                          ParseError(Error{ pos: 1, msg: "Expecting '0-9 or e' but got 'd'".to_owned() })]);
    }

    #[test]
    fn parse_error_on_unmatched_value_ending() {
        let msg = "Unmatched value ending".to_owned();
        assert_stream_eq("e", [ParseError(Error{ pos: 1, msg: msg.clone() })]);
        assert_stream_eq("lee", [ListStart, ListEnd, ParseError(Error{ pos: 3, msg: msg.clone() })]);
    }

}

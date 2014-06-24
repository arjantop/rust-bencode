RUSTC = rustc
RUSTDOC = rustdoc
RUSTFLAGS = -O
BUILDDIR = target
TESTDIR = $(BUILDDIR)/test

all: $(BUILDDIR) test lib docs

$(BUILDDIR):
	mkdir -p $@

$(TESTDIR): $(BUILDDIR)
	mkdir -p $@

lib:
	cargo build

clean:
	rm -rf $(BUILDDIR)

test: libtest doctest

libtest: $(TESTDIR)
	$(RUSTC) --test -o $(TESTDIR)/test src/bencode.rs
	RUST_LOG=std::rt::backtrace ./$(TESTDIR)/test

bench: $(TESTDIR)
	$(RUSTC) $(RUSTFLAGS) --test -o $(TESTDIR)/bench src/bencode.rs
	./$(TESTDIR)/bench --bench

doctest: lib
	$(RUSTDOC) -L $(BUILDDIR) --test src/bencode.rs

docs:
	$(RUSTDOC) src/bencode.rs

.PHONY: lib clean test libtest bench doctest docs

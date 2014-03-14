RUSTC = rustc
RUSTFLAGS = -O
BUILDDIR = build
TESTDIR = $(BUILDDIR)/test

all: $(BUILDDIR) test lib

$(BUILDDIR):
	mkdir -p $@

$(TESTDIR): $(BUILDDIR)
	mkdir -p $@

lib:
	${RUSTC} ${RUSTFLAGS} --out-dir $(BUILDDIR) src/bencode.rs

clean:
	rm -rf $(BUILDDIR)

test: $(TESTDIR)
	${RUSTC} ${RUSTFLAGS} --test -o $(TESTDIR)/test src/bencode.rs
	RUST_LOG=std::rt::backtrace ./$(TESTDIR)/test

bench: test
	./$(TESTDIR)/test --bench

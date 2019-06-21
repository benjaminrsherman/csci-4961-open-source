all: dynamic_block static_block

dynamic_block: program.o sharedlib.so
	cc -o dynamic_block program.o sharedlib.so -Wl,-rpath='$$ORIGIN'

static_block: program.o staticlib.a
	cc -o static_block program.o staticlib.a

program.o: program.c
	cc -c program.c

block.o: source/block.c
	cc -c source/block.c

sharedlib.so: block.o
	cc -shared -o sharedlib.so block.o

staticlib.a: block.o
	ar qc staticlib.a block.o

clean:
	rm -f dynamic_block static_block *.o *.so *.a

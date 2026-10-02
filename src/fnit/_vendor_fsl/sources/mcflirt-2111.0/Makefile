include ${FSLCONFDIR}/default.mk

PROJNAME = mcflirt
XFILES   = mcflirt
LIBS     = -lfsl-newimage -lfsl-miscmaths -lfsl-utils \
           -lfsl-NewNifti -lfsl-znz -lfsl-cprob

all: ${XFILES}

mcflirt: mcflirt.o Globaloptions.o Log.o
	${CXX} ${CXXFLAGS} -o $@  $^ ${LDFLAGS}

mcflirt2: mcflirt2.o
	${CXX} ${CXXFLAGS} -o $@  $^ ${LDFLAGS}

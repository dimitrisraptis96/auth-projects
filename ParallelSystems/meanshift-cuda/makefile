CC=nvcc

SRCDIR=src
BINDIR=bin
BUILDDIR=build
TARGET=bin/meanshift

SRCEXT := c
SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT)*)
OBJECTS := $(SOURCES:.c*=.o)

NVFLAGS=-g -G -04 -lm -lcuda -lcudart
INC := -I include

RM = rm -f

$(TARGET): $(OBJECTS)
	@mkdir -p $(BINDIR)
	$(CC) $^ -o $(TARGET) -lm -g -G

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)*
	@mkdir -p $(BUILDDIR)
	$(CC) $(NVFLAGS) $(INC) -c -o $@ $<

clean:
	$(RM) -r $(BUILDDIR) $(TARGET)

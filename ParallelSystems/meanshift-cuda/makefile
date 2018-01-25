CC=nvcc

SRCDIR=src
BUILDDIR=build
TARGET=bin/main

SRCEXT := c
SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT)*)
# OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)*=.o))
OBJECTS := $(SOURCES:.c*=.o)

CFLAGS=-g -O4 -Wall -lm
INC := -I include

RM = rm -f

$(TARGET): $(OBJECTS)
	$(CC) $^ -o $(TARGET) -lm

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	  @mkdir -p $(BUILDDIR)
	  $(CC) $(CFLAGS) $(INC) -c -o $@ $<

clean:
	$(RM) -r $(BUILDDIR) $(TARGET)
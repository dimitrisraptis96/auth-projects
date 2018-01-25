CC=nvcc

SRCDIR=src
BUILDDIR=build
TARGET=bin/main

SRCEXT := c
SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT)*)
# OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)*=.o))
OBJECTS := $(SOURCES:.c*=.o)

CFLAGS=-g -O4 -Wall -lm 
NVFLAGS=-g -04 -lm -lcuda -lcudart
INC := -I include

RM = rm -f

$(TARGET): $(OBJECTS)
	$(CC) $^ -o $(TARGET) -lm

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)*
	  @mkdir -p $(BUILDDIR)
	  $(CC) $(NVFLAGS) $(INC) -c -o $@ $<

clean:
	$(RM) -r $(BUILDDIR) $(TARGET)
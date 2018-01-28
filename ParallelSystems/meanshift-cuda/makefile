CC=nvcc

SRCDIR=src
BUILDDIR=build
TARGET=bin/meanshift

SRCEXT := c
SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT)*)
OBJECTS := $(SOURCES:.c*=.o)
 
NVFLAGS=-g -04 -lm -lcuda -lcudart
INC := -I include

RM = rm -f

$(TARGET): $(OBJECTS)
	$(CC) $^ -o $(TARGET) -lm --gpu-architecture=compute_50 --gpu-code=compute_50,sm_50

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)*
	  @mkdir -p $(BUILDDIR)
	  $(CC) $(NVFLAGS) $(INC) -c -o $@ $<

clean:
	$(RM) -r $(BUILDDIR) $(TARGET)
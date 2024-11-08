# Compiler
CC = gcc

# Compiler flags
CFLAGS = -I../include -I/usr/include/hdf5 -I/usr/include/openblas -I/usr/include -lmatio -lhdf5

# Linker flags (move the libraries to LDFLAGS)
LDFLAGS = -lmatio -lhdf5 -lopenblas -lm

# Source files
SRCS = main.c random_input.c print_matrix.c calculate_distance.c quick_select.c find.c truncate.c load.c

# Object files (replace .c with .o)
OBJS = $(SRCS:.c=.o)

# Target executable
TARGET = kNN

# Default rule
all: $(TARGET)

# Link object files to create the executable
$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

# Compile source files to object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up generated files
clean:
	rm -f $(OBJS) $(TARGET)

# Specify phony targets
.PHONY: all clean


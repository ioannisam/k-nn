# Compiler
CC = gcc

# Compiler flags
CFLAGS = -I../include -I/usr/include/openblas -fopenmp

# Source files
SRCS = main.c random_input.c print_matrix.c calculate_distance.c

# Object files (replace .c with .o)
OBJS = $(SRCS:.c=.o)

# Target executable
TARGET = kNN

# Default rule
all: $(TARGET)

# Link object files to create the executable
$(TARGET): $(OBJS)
	$(CC) -o $@ $^ -lopenblas -fopenmp  # Add -fopenmp here

# Compile source files to object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up generated files
clean:
	rm -f $(OBJS) $(TARGET)

# Specify phony targets
.PHONY: all clean

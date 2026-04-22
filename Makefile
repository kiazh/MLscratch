CC      = clang
CFLAGS  = -Wall -Wextra -O2
TARGET  = src/main

all: $(TARGET)

$(TARGET): src/main.c
	$(CC) $(CFLAGS) -o $(TARGET) src/main.c -lm

clean:
	rm -f $(TARGET)

.PHONY: all clean

CC = gcc
CFLAGS = -O3 -Wall -fPIC
LDFLAGS = -shared

TARGET = libstereo.so
SRC = src/stereo_core.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)

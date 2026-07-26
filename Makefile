# Convenience wrapper. The real build lives in host/Makefile and west.
.PHONY: test test-arm size fixtures golden clean firmware

test:      ; $(MAKE) -C host test
test-arm:  ; $(MAKE) -C host test-arm
size:      ; $(MAKE) -C host size-m7
fixtures:  ; $(MAKE) -C host fixtures
golden:    ; $(MAKE) -C host golden
clean:     ; $(MAKE) -C host clean

firmware:
	west build -b teensy41 -p always firmware/teensy41-tinyllm

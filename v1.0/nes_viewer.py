#!/usr/bin/env python3
"""
nes_viewer.py
Simple iNES ROM loader + CHR tile viewer + skeleton CPU/PPU mapping.

Usage:
    python nes_viewer.py path/to/rom.nes
"""

import sys
import os
import struct
import pygame
from pygame.locals import *
from math import ceil

# -------------------------
# iNES parsing & ROM loader
# -------------------------

class INESRom:
    def __init__(self, path):
        self.path = path
        self.prg_rom = b""
        self.chr_rom = b""
        self.trainer = None
        self.mapper = 0
        self.mirroring = "horizontal"
        self.battery = False
        self.four_screen = False
        self.prg_banks = 0
        self.chr_banks = 0
        self.format = "iNES"
        self._load()

    def _load(self):
        with open(self.path, "rb") as f:
            header = f.read(16)
            if len(header) < 16:
                raise ValueError("Not a valid .nes file (too short)")
            if header[0:4] != b"NES\x1a":
                raise ValueError("Not iNES: missing NES<EOF> signature")
            (prg_banks, chr_banks, flags6, flags7, prg_ram, flags9, flags10) = struct.unpack("<BBBBBBB", header[4:11])
            self.prg_banks = prg_banks
            self.chr_banks = chr_banks
            mapper_low = (flags6 >> 4)
            mapper_high = (flags7 >> 4)
            self.mapper = mapper_low | (mapper_high << 4)
            # mirroring etc
            self.mirroring = "vertical" if (flags6 & 0x01) else "horizontal"
            self.battery = bool(flags6 & 0x02)
            self.trainer_present = bool(flags6 & 0x04)
            self.four_screen = bool(flags6 & 0x08)
            # check trainer
            if self.trainer_present:
                self.trainer = f.read(512)
            # read PRG-ROM
            prg_size = prg_banks * 16 * 1024
            self.prg_rom = f.read(prg_size)
            # read CHR-ROM (if 0 banks -> CHR RAM)
            chr_size = chr_banks * 8 * 1024
            self.chr_rom = f.read(chr_size) if chr_size > 0 else b""
            # note: this loader does not parse NES 2.0 extended headers

    def info_str(self):
        return f"ROM: {os.path.basename(self.path)} | PRG: {self.prg_banks*16}KB | CHR: {self.chr_banks*8}KB | mapper: {self.mapper} | mirroring: {self.mirroring} | trainer: {self.trainer_present}"

# -------------------------
# CHR tile decoding
# -------------------------
# NES CHR tiles: each tile is 16 bytes:
# - first 8 bytes are bitplane 0 (one byte per row)
# - next 8 bytes are bitplane 1 (one byte per row)
# Pixel color index = (bit1 << 1) | bit0 -> 0..3

def decode_tile(tile_bytes):
    """Given 16 bytes, return an 8x8 matrix of values 0..3"""
    assert len(tile_bytes) == 16
    plane0 = tile_bytes[:8]
    plane1 = tile_bytes[8:16]
    pixels = [[0]*8 for _ in range(8)]
    for y in range(8):
        b0 = plane0[y]
        b1 = plane1[y]
        for x in range(8):
            bit = 7 - x
            v0 = (b0 >> bit) & 1
            v1 = (b1 >> bit) & 1
            pixels[y][x] = (v1 << 1) | v0
    return pixels

# simplified NES palette (RGB tuples) — this is not the real hardware palette but useful for viewing.
NES_PALETTE = [
    (84, 84, 84),
    (0, 30, 116),
    (8, 16, 144),
    (48, 0, 136),
    (68, 0, 100),
    (92, 0, 48),
    (84, 4, 0),
    (60, 24, 0),
    (32, 42, 0),
    (8, 58, 0),
    (0, 64, 0),
    (0, 60, 0),
    (0, 50, 60),
    (0, 0, 0),
    (0, 0, 0),
    (0, 0, 0),
]
# We'll map palette indices 0..3 to simple colors (for CHR viewing)
VIEW_PALETTE = [
    (84,84,84),
    (200, 100, 60),
    (120, 200, 140),
    (240, 240, 120)
]

# -------------------------
# Simple PPU viewer (renders CHR tiles)
# -------------------------
class CHRViewer:
    def __init__(self, ines_rom, scale=2, tiles_per_row=16):
        self.rom = ines_rom
        self.scale = scale
        self.tiles_per_row = tiles_per_row
        self.tiles = []
        if len(self.rom.chr_rom) == 0:
            print("No CHR ROM present (CHR RAM). Nothing to display.")
        else:
            self._decode_all_tiles()

    def _decode_all_tiles(self):
        data = self.rom.chr_rom
        tile_count = len(data) // 16
        self.tiles = []
        for i in range(tile_count):
            tile_bytes = data[i*16:(i+1)*16]
            pixels = decode_tile(tile_bytes)
            self.tiles.append(pixels)

    def render_surface(self):
        if not self.tiles:
            return None
        tpr = self.tiles_per_row
        rows = ceil(len(self.tiles) / tpr)
        tile_w = 8 * self.scale
        tile_h = 8 * self.scale
        surf = pygame.Surface((tpr * tile_w, rows * tile_h))
        for idx, tile in enumerate(self.tiles):
            tx = (idx % tpr) * tile_w
            ty = (idx // tpr) * tile_h
            # draw tile
            for y in range(8):
                for x in range(8):
                    v = tile[y][x]  # 0..3
                    color = VIEW_PALETTE[v]
                    rect = pygame.Rect(tx + x*self.scale, ty + y*self.scale, self.scale, self.scale)
                    surf.fill(color, rect)
        return surf

# -------------------------
# CPU / PPU skeletons (start here to implement full emulator)
# -------------------------

class CPU6502:
    def __init__(self):
        # registers
        self.A = 0
        self.X = 0
        self.Y = 0
        self.PC = 0
        self.SP = 0xFD
        self.P = 0x24  # status
        # memory: 64KB
        self.mem = bytearray(0x10000)

    def reset(self, rom: INESRom):
        # basic mapper 0, map first PRG bank at 0x8000, last bank at 0xC000 (if 16KB PRG) or whole if 32KB
        if rom.prg_banks == 1:
            # 16KB -> mirrored
            self.mem[0x8000:0xC000] = rom.prg_rom
            self.mem[0xC000:0x10000] = rom.prg_rom
        else:
            # 32KB -> load as-is
            self.mem[0x8000:0x8000+len(rom.prg_rom)] = rom.prg_rom
        # set reset vector from 0xFFFC
        self.PC = self.read_word(0xFFFC)

    def read_word(self, addr):
        lo = self.mem[addr & 0xFFFF]
        hi = self.mem[(addr+1) & 0xFFFF]
        return lo | (hi << 8)

    # Minimal step stub (not implemented)
    def step(self):
        opcode = self.mem[self.PC]
        # For now, just increment PC so CPU doesn't get stuck. Real implementation required.
        self.PC = (self.PC + 1) & 0xFFFF
        return opcode

class PPU:
    def __init__(self):
        # PPU has its own memory map, pattern tables come from CHR ROM
        pass

# -------------------------
# Main visual UI using pygame
# -------------------------
def main(rom_path):
    rom = INESRom(rom_path)
    print(rom.info_str())
    viewer = CHRViewer(rom, scale=3, tiles_per_row=16)

    pygame.init()
    pygame.display.set_caption(f"NES CHR Viewer - {os.path.basename(rom_path)}")
    if viewer.tiles:
        surf = viewer.render_surface()
        w, h = surf.get_size()
    else:
        w, h = 320, 240
        surf = pygame.Surface((w, h))
        surf.fill((20,20,20))
    # make window resizable
    screen = pygame.display.set_mode((w, h), RESIZABLE)
    clock = pygame.time.Clock()
    running = True
    font = pygame.font.SysFont("monospace", 14)

    while running:
        for ev in pygame.event.get():
            if ev.type == QUIT:
                running = False
            elif ev.type == VIDEORESIZE:
                screen = pygame.display.set_mode((ev.w, ev.h), RESIZABLE)
            elif ev.type == KEYDOWN:
                if ev.key == K_ESCAPE:
                    running = False

        screen.fill((40, 40, 40))
        # blit the tile surface centered and scaled to window while preserving pixel-art scale
        if viewer.tiles:
            # we will draw at original size; user can resize window.
            screen.blit(surf, (0,0))
        else:
            screen.blit(surf, (0,0))
            txt = font.render("No CHR ROM (CHR RAM) or no tiles to display.", True, (220,220,220))
            screen.blit(txt, (10,10))

        # HUD text
        info = f"{os.path.basename(rom.path)} | PRG: {rom.prg_banks*16}KB | CHR: {rom.chr_banks*8}KB | mapper {rom.mapper}"
        hud = font.render(info, True, (240,240,240))
        screen.blit(hud, (8, screen.get_height()-20))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python nes_viewer.py path/to/rom.nes")
        sys.exit(1)
    rom_path = sys.argv[1]
    if not os.path.exists(rom_path):
        print("File not found:", rom_path)
        sys.exit(1)
    main(rom_path)

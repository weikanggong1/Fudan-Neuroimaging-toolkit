"""Own minimal lossless RGB8 PNG encoder for saved-label figures; stdlib only."""
import struct
import zlib


def encode_rgb_png(width,height,pixels,annotation):
    if not (type(width) is int and type(height) is int and 1<=width<=16384 and 1<=height<=16384):
        raise ValueError("positive bounded image geometry required")
    if type(pixels) is not bytes or len(pixels)!=width*height*3:
        raise ValueError("contiguous RGB8 byte length mismatch")
    if type(annotation) is not bytes or b"\x00" in annotation:
        raise ValueError("ASCII PNG annotation without null required")
    annotation.decode("ascii")
    def chunk(kind,payload):
        return struct.pack(">I",len(payload))+kind+payload+struct.pack(">I",zlib.crc32(kind+payload)&0xffffffff)
    header=struct.pack(">IIBBBBB",width,height,8,2,0,0,0)
    stride=width*3
    rows=b"".join(b"\x00"+pixels[start:start+stride] for start in range(0,len(pixels),stride))
    return (b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",header)+chunk(b"tEXt",b"FNIT source\x00"+annotation)
        +chunk(b"IDAT",zlib.compress(rows,9))+chunk(b"IEND",b""))

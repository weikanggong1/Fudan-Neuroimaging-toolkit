"""Tiny PNG transport contracts only; not an MRI/scientific benchmark."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import zlib
from png_encoder import encode_rgb_png


def main():
    rows=[]
    for width,height in ((1,1),(2,3),(3,2),(5,7)):
        pixels=bytes((i*37)%256 for i in range(width*height*3))
        annotation=b'{"rows":["old","candidate","official"],"contract":true}'
        png=encode_rgb_png(width,height,pixels,annotation)
        assert png[:8]==b"\x89PNG\r\n\x1a\n"
        offset=8;chunks=[]
        while offset<len(png):
            length=struct.unpack(">I",png[offset:offset+4])[0];kind=png[offset+4:offset+8]
            data=png[offset+8:offset+8+length];crc=struct.unpack(">I",png[offset+8+length:offset+12+length])[0]
            assert zlib.crc32(kind+data)&0xffffffff==crc
            chunks.append((kind,data));offset+=length+12
        assert offset==len(png) and [k for k,d in chunks]==[b"IHDR",b"tEXt",b"IDAT",b"IEND"]
        assert struct.unpack(">IIBBBBB",chunks[0][1])==(width,height,8,2,0,0,0)
        assert chunks[1][1]==b"FNIT source\x00"+annotation
        data=zlib.decompress(chunks[2][1]);stride=width*3
        assert len(data)==height*(stride+1)
        restored=b"".join(data[y*(stride+1)+1:(y+1)*(stride+1)] for y in range(height))
        assert all(data[y*(stride+1)]==0 for y in range(height)) and restored==pixels
        rows.append({"width":width,"height":height,"CRC_and_RGB_bytes_exact":True})
    negative=0
    for width,height,pixels,annotation in ((0,1,b"",b""),(2,3,b"short",b""),(1,1,b"rgb",b"bad\x00")):
        try:encode_rgb_png(width,height,pixels,annotation)
        except ValueError:negative+=1
        else:raise RuntimeError("invalid PNG contract accepted")
    leaf=Path(__file__).resolve().parent
    assert "torch" not in sys.modules and "numpy" not in sys.modules
    report={"schema":"fnit_owned_png_small_contracts/v1","status":"passed",
        "positive_cases":rows,"negative_cases":negative,"Torch_or_NumPy_imported":False,
        "MRI_native_model_GPU_calls":0,"scope":"RGB8 transport/CRC/annotation and bounded geometry; synthetic tiny pixels only, not performance benchmark",
        "source_sha256":{name:hashlib.sha256((leaf/name).read_bytes()).hexdigest() for name in
            ("png_encoder.py","check_png_contracts.py","score_saved_whole_v2.py")}}
    (leaf/"PNG_CONTRACTS.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":"passed","PNG_positive":4,"negative":negative,"MRI_calls":0}))


if __name__=="__main__":main()

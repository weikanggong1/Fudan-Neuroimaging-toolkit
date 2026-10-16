"""Audit tracked publication files against a private, user-supplied regex policy.

The JSON policy is a list of regular-expression strings stored outside the
repository. Reports contain file identities and counts, never matched text or
policy values. Images are checked through textual metadata; compressed pixels
are not treated as text. This command reads files and does not redact them.
"""
from __future__ import annotations

import argparse
import gzip
import io
import hashlib
import json
import marshal
from pathlib import Path
import re
import struct
import subprocess
import tarfile
import zlib


def _png_metadata(raw):
    position=8
    while position+12 <= len(raw):
        size=struct.unpack('>I',raw[position:position+4])[0]
        kind=raw[position+4:position+8]
        body=raw[position+8:position+8+size]
        if kind==b'tEXt':
            yield body.decode('latin-1')
        elif kind==b'eXIf':
            yield body.decode('latin-1')
        elif kind==b'zTXt':
            _,remainder=body.split(b'\0',1)
            yield zlib.decompress(remainder[1:]).decode('latin-1')
        elif kind==b'iTXt':
            _,remainder=body.split(b'\0',1)
            flag=remainder[0]
            _,_,data=remainder[2:].split(b'\0',2)
            yield (zlib.decompress(data) if flag else data).decode('utf-8')
        position += size+12


def _gzip_metadata(raw):
    if len(raw)<10 or raw[:3] != b'\x1f\x8b\x08':return
    flags=raw[3];position=10
    if flags&4:
        size=struct.unpack('<H',raw[position:position+2])[0]
        position+=2+size
    for flag in (8,16):
        if flags&flag:
            end=raw.index(b'\0',position)
            yield raw[position:end].decode('latin-1')
            position=end+1


def audit(repository,patterns,*,include_untracked=False):
    """Return counts and opaque identities; regex policy stays with the caller."""
    repository=Path(repository).resolve()
    arguments=['git','-C',str(repository),'ls-files','--cached','-z']
    if include_untracked:arguments[4:4]=['--others','--exclude-standard']
    names=subprocess.check_output(arguments).decode('utf-8').split('\0')
    compiled=[re.compile(pattern,re.I) for pattern in patterns]
    records=[];count=0;binary_skipped=0
    for name in sorted(set(names)-{''}):
        path=repository/name
        if not path.is_file():continue
        count+=1
        raw=path.read_bytes()
        texts=[name]
        texts.extend(_gzip_metadata(raw))
        if raw.startswith(b'\x89PNG\r\n\x1a\n'):
            texts.extend(_png_metadata(raw))
        elif name.endswith(('.tar.gz','.tgz','.tar')):
            with tarfile.open(fileobj=io.BytesIO(raw),mode='r:*') as archive:
                for member in archive:
                    texts.extend((member.name,member.linkname,member.uname,member.gname))
                    texts.extend(str(value) for value in member.pax_headers.values())
                    if member.isfile():
                        payload=archive.extractfile(member).read()
                        texts.extend(_gzip_metadata(payload))
                        if payload.startswith(b'\x89PNG\r\n\x1a\n'):
                            texts.extend(_png_metadata(payload))
                        elif b'\0' not in payload:
                            try:texts.append(payload.decode('utf-8'))
                            except UnicodeError:pass
        elif name.endswith(('.log.gz','.txt.gz','.json.gz')):

            texts.append(gzip.decompress(raw).decode('utf-8'))
        elif path.suffix in ('.pstats','.prof'):
            try:
                profile=marshal.loads(raw)
                def strings(item):
                    if isinstance(item,str):yield item
                    elif isinstance(item,dict):
                        for key,value in item.items():yield from strings(key);yield from strings(value)
                    elif isinstance(item,(tuple,list,set,frozenset)):
                        for value in item:yield from strings(value)
                texts.extend(strings(profile))
            except (ValueError,EOFError):
                try:texts.append(raw.decode('utf-8'))
                except UnicodeError:binary_skipped+=1
        else:
            try:
                text=raw.decode('utf-8')
                if '\0' in text:binary_skipped+=1
                else:texts.append(text)
            except UnicodeError:
                binary_skipped+=1
        findings={f'pattern_{index}':sum(len(pattern.findall(text)) for text in texts)
                  for index,pattern in enumerate(compiled)}
        findings={key:value for key,value in findings.items() if value}
        if findings:
            records.append({'file_id':hashlib.sha256(name.encode()).hexdigest(),
                            'file_sha256':hashlib.sha256(raw).hexdigest(),
                            'matches':findings})
    return {'schema_version':1,'files_scanned':count,
            'binary_files_without_text_metadata':binary_skipped,
            'matching_files':len(records),'records':records,
            'policy_values_in_report':False,'matched_text_in_report':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository',type=Path,default=Path.cwd())
    parser.add_argument('--policy',type=Path,required=True,
                        help='private JSON list of regex patterns; do not commit this file')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--include-untracked',action='store_true')
    options=parser.parse_args(argv)
    patterns=json.loads(options.policy.read_text())
    if not isinstance(patterns,list) or not patterns or any(not isinstance(x,str) or not x for x in patterns):
        parser.error('policy must be a nonempty JSON list of regex strings')
    report=audit(options.repository,patterns,include_untracked=options.include_untracked)
    options.output.parent.mkdir(parents=True,exist_ok=True)
    options.output.write_text(json.dumps(report,indent=2)+'\n')
    return int(report['matching_files'] != 0)


if __name__=='__main__':raise SystemExit(main())

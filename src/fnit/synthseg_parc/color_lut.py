"""Write an explicitly supplied FreeSurfer text LUT using nibabel only.

The NIfTI ecode-14 format is documented by FreeSurfer's tagged extension:
big-endian intent/version, tag 1 with a version-2 color table, and end tag -1.
No FreeSurfer or Surfa runtime is imported.
"""

from pathlib import Path
import struct

import nibabel as nib


def color_lut_extension(path: str | Path) -> nib.nifti1.Nifti1Extension:
    """Read ``id name R G B T`` text rows and return an ecode-14 extension.

    ``T`` is the 0–255 FreeSurfer transparency byte, with zero opaque. Blank
    lines and ``#`` comments are ignored; malformed/duplicate entries raise
    ValueError. Names are encoded as UTF-8 and terminated by a NUL byte.
    """
    entries = {}
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        columns = line.split("#", 1)[0].split()
        if not columns:
            continue
        if len(columns) != 6:
            raise ValueError(f"Invalid color LUT row {line_number}: expected six columns")
        try:
            label = int(columns[0])
            color = tuple(int(v) for v in columns[2:])
        except ValueError as error:
            raise ValueError(f"Invalid numeric color LUT row {line_number}") from error
        if not 0 <= label < 2**31 - 1 or any(not 0 <= v <= 255 for v in color):
            raise ValueError(f"Color LUT values outside valid ranges on row {line_number}")
        if label in entries:
            raise ValueError(f"Duplicate color LUT label {label}")
        entries[label] = (columns[1], color)
    if not entries:
        raise ValueError("Color LUT contains no entries")
    table = bytearray(struct.pack(">iiii", -2, max(entries) + 1, 0, len(entries)))
    for label, (name, color) in entries.items():
        encoded_name = name.encode("utf-8") + b"\x00"
        table.extend(struct.pack(">ii", label, len(encoded_name)))
        table.extend(encoded_name)
        table.extend(struct.pack(">iiii", *color))
    content = (b">\x00\x00\x01" + struct.pack(">iq", 1, len(table)) + table
               + struct.pack(">iqi", 7, 4, 1) + struct.pack(">iq", -1, 1) + b"*")
    return nib.nifti1.Nifti1Extension(14, content)


def attach_color_lut(image: nib.Nifti1Image, path: str | Path) -> None:
    """Attach the explicit LUT to a NIfTI header, replacing prior ecode 14."""
    extension = color_lut_extension(path)
    retained = [value for value in image.header.extensions if value.get_code() != 14]
    image.header.extensions.clear()
    image.header.extensions.extend(retained)
    image.header.extensions.append(extension)
    image.extra["color_lut"] = str(Path(path))

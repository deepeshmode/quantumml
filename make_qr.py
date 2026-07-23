"""
QR code for the poster.

Usage:
    python make_qr.py <url> [outfile.png]

Sized for print: high error correction so it still scans if the poster gets
scuffed or the print is slightly off-register, and large enough that a phone
locks on from a normal standing distance. At 25-30 mm printed, keep the URL
short - long URLs force a denser grid that needs a bigger square to scan.
"""

import sys
import qrcode
from qrcode.constants import ERROR_CORRECT_H


def make(url, outfile="qr_repo.png", box_size=20, border=3):
    qr = qrcode.QRCode(
        version=None,              # auto-fit to the URL length
        error_correction=ERROR_CORRECT_H,   # ~30% recoverable
        box_size=box_size,
        border=border,             # quiet zone; below 4 modules risks scan failure
    )
    qr.add_data(url)
    qr.make(fit=True)

    img = qr.make_image(fill_color="#0b0b0b", back_color="#fcfcfb")
    img.save(outfile)

    px = img.size[0]
    print(f"{outfile}  {px}x{px}px  version {qr.version}  url: {url}")
    print(f"  at 300 dpi that prints {px/300*25.4:.0f} mm square")
    return outfile


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    make(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "qr_repo.png")

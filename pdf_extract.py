# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 12:28:03 2016
Extract text from pdfs.

Dependencies:
- pdfminer
- tesseract-ocr (not Python)
- pyocr
- wand
- imagemagick
- PIL
- ghostscript (not Python)
- jpeg (not Python)

@author: salo
"""
import os
from glob import glob
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from cStringIO import StringIO
import subprocess


def convert_pdf_to_txt(path):
    """
    From top answer here:
    http://stackoverflow.com/questions/26494211/extracting-text-from-a-pdf-file-using-pdfminer-in-python
    """
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = file(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ''
    maxpages = 0
    caching = True
    pagenos = set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages,
                                  password=password, caching=caching,
                                  check_extractable=False):
        # Check for, and skip, pages that are mostly image, I think.
        try:
            interpreter.process_page(page)
        except:
            print(path)

    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()
    return text


def convert_pdfs(files, out_dir):
    """ Loop through a list of pdf files, pull text for each and write to txt
    file in out_dir.
    """
    image_based = []
    for f in files:
        # Check pdf fonts to determine if it is text- or image-based
        pmid = os.path.splitext(os.path.basename(f))[0]
        if not os.path.isfile(os.path.join(out_dir, '{0}.txt'.format(pmid))):
            txt = convert_pdf_to_txt(f)
            with open(os.path.join(out_dir, '{0}.txt'.format(pmid)), 'wb') as fo:
                fo.write(txt)
    return image_based


def test():
    files = glob('/Users/salo/Google Drive/ATHENA-pdfs/QC-Text-based-pdfs/*.pdf')
    out_dir = '/Users/salo/Desktop/athena-text/'
    image_based = convert_pdfs(files, out_dir)
    return image_based

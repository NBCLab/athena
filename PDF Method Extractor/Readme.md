Most of the documentation is found within the .java source code.

This program uses pdfbox 1.8.10 -- it is included in the executable jar.

In short, you select a directory that contains pdfs at some level.  The program does search through subfolders.
When it finishes, a dialog box appears saying "Done"

The output folder appears wherever the jar was run "./ExtractMethod Output"
There is one output text file for every successful extraction -- it keeps the name of the file. 
There is also a file that documents the failures "Failed.txt"
It also lists 1 of 2 reasons for the failure:
	"Could not find the end of the methods" -- meaning that there was not a valid header after the method section.
	or
	"Could not find the methods" -- meaning that there was not a valid method header.

I improved the methods extraction code.  Before, it only grabbed text between various forms of the Method and the Results headers.

Now, it searches for a valid Method (Materials and Method, 2. Patients and methods, etc) header followed by a valid header (like Results, Conclusion, Acknowledgments, or References).  This was necessary to fix issues where the Method section was out of order.

The non-Method header search criteria is also more generic and flexible than before.  This was necessary to select headers that did not have an overarching results section and split it into sections like "fMRI results" and "Behavioral results."

The program still cannot extract text from documents that
-have certain security settings which prevent text extraction.
-treat the text from two columns as a single line.
-have too few [or a lack of] headers (one article has the Method at the end without headers for the references or acknowledgments).
-have 'unique' headers (one article uses "Experimental Procedure" instead of "Method").
-have garbled text due to poor OCR processing.
-and, more obviously, have scanned pictures of text instead of text.

I had to rename the executable .jar file as a .txt file so gmail would take it.
I included the source .java file, as well -- it is fairly heavily commented.
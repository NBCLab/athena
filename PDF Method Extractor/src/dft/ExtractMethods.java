// PDF Extract Methods
// v0.33
// Jason Hays
// This program uses Apache PDFBox 1.8.10 to extract the Methods section using regular expressions.
//
// It assumes that the Method section will be followed by another important header:
// 		Results, Discussion, Conclusion, Acknowledgments, References. 
// The Method section's name can vary somewhat in casing and what accompanies it.
// The output uses the name of the pdf to create a .txt file in the "ExtractMethods Output" folder.
//  Each line in the output has a methods section (if there are multiple experiments).

package dft;

import java.awt.Frame;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Scanner;
import java.util.Vector;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.DataFormatException;

import javax.swing.JFileChooser;
import javax.swing.JOptionPane;
import javax.xml.bind.DataBindingException;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDDocumentInformation;
import org.apache.pdfbox.util.PDFTextStripper;

public class ExtractMethods {
	// these are the regular expressions used to find dashes and periods in the methods.
	// this has to be a dash followed by newline related characters or the end of the text.
	static Pattern dashKiller = Pattern.compile("(-[\r\n]+|\\z)");
	// remove all new lines with this.
	static Pattern newLineKiller = Pattern.compile("([\r\n]+|\\z)");

	// remove all cases of text matching Pattern 'p' from the text.
	// this function is used to clean up the extracted text.
	// article pdf text has an unnecessarily large number of new lines as well as 
	//  words split because of dashes.
	public static String removePatternInstances(String text, Pattern p, String newSeparator) {
		String[] groups = p.split(text);

		// the corrected text is stored here
		String fixedText = "";
		// go through each string "str" within groups
		for (String str : groups) {
			// concatenate the strings
			fixedText += newSeparator+str;
		}
		return fixedText;
	}

	// this function removes dashes at the end of lines (separating words) and concatenates lines.
	public static String cleanup(String section) {
		// dashes need to be handled first because newline characters are what separate
		// bad dashes from good dashes.
		section = removePatternInstances(section, dashKiller, "");

		section = removePatternInstances(section, newLineKiller, " ");

		// return the cleaned text.
		return section;
	}
	// get all of the pdfs in the directory specified.
	public static void getFiles(File dir, Vector<File> pdfFiles, Vector<File> textFiles) {
		// the list of all files (including non pdfs)
		File[] fList = dir.listFiles();

		// go through the files one by one.
		for (File f : fList) {

			if (f.isFile()) {
				// process files that have pdf extensions (add them to the list)
				if (f.getName().endsWith(".pdf"))
					pdfFiles.add(f);
				else if (f.getName().endsWith(".txt")) {
					textFiles.add(f);
				}

				// process the contents of sub directories too.
			} else if (f.isDirectory()) {
				getFiles(f, pdfFiles, textFiles);
			}
		}
	}
	// the main method is where the program starts
	public static void main(String args[]) {
		// increase the buffer size
		System.setProperty("org.apache.pdfbox.baseParser.pushBackSize", "999000");
		// make a window for the file chooser
		Frame f = new Frame();
		JFileChooser fc = new JFileChooser();

		// start the program wherever the program is located.
		fc.setCurrentDirectory(new File("./"));
		// only allow directories to be chosen
		fc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);

		// open the dialog (not using tmp)
		int tmp = fc.showOpenDialog(f);
		if (tmp == JFileChooser.CANCEL_OPTION)
			return;
		// get the folder they chose
		File dirName = fc.getSelectedFile();
		// dispose of the window.
		f.dispose();

		String saveDir = "ExtractMethods Output/";
		File outDir = new File(saveDir);
		outDir.mkdir();
		// store which documents could not have the methods extracted by this.
		File failF = new File(saveDir+"Failed.txt");
		FileWriter fw = null;

		try {
			if (!failF.exists())
				failF.createNewFile();
			fw = new FileWriter(failF);
		} catch (IOException e1) {
			e1.printStackTrace();
		}

		File fileNames = new File(saveDir+"Names.txt");
		FileWriter nw = null;

		try {
			if (!fileNames.exists())
				fileNames.createNewFile();
			nw = new FileWriter(fileNames);
		} catch (IOException e1) {
			e1.printStackTrace();
		}

		// initialize the list of pdf files in the directory
		Vector<File> pdfList = new Vector<File>();
		Vector<File> txtList = new Vector<File>();
		// get the files.
		getFiles(dirName, pdfList, txtList);
		// get the names
		for (File pdf : pdfList) {
			try {
				nw.write(pdf.getName()+'\n');
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		for (File txt : txtList) {
			try {
				nw.write(txt.getName()+'\n');
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		try {
			nw.close();
		} catch (IOException e1) {
			e1.printStackTrace();
		}
		// go through each pdf found earlier
		for (File pdf : pdfList) {
			//System.out.println(pdf.getName());
			PDDocument pddDocument = null;
			try {
				// load the contents as a PDF rather than a generic File
				pddDocument = PDDocument.load(pdf);
			} catch (IOException e) {
				e.printStackTrace();
			}

			// set up the text extractor
			PDFTextStripper stripper = null;
			try {
				stripper = new PDFTextStripper();
			} catch (IOException e) {
				e.printStackTrace();
			}

			// set it to pull text from the entire document
			stripper.setPageStart("1");
			stripper.setPageEnd(Integer.toString(pddDocument.getNumberOfPages()));

			// pull the text and save it as "text"
			String text = null;
			try {
				text = stripper.getText(pddDocument);
			} catch (IOException e) {
				e.printStackTrace();
			} catch (NullPointerException e) {
				e.printStackTrace();
			}
			// close the pdf
			try {
				pddDocument.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
			processFileText(text, fw, pdf, saveDir);
		}
		for (File txt : txtList) {
			Scanner stripper = null;
			try {
				stripper = new Scanner(txt);
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}
			
			String text = null;
			try {
				text = new String(Files.readAllBytes(Paths.get(txt.getAbsolutePath())), StandardCharsets.UTF_8);
			} catch (IOException e) {
				e.printStackTrace();
			}

			processFileText(text, fw, txt, saveDir);
		}
		try {
			// make sure the file writer (for failed files) finished outputting.
			fw.flush();
			fw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		JOptionPane.showMessageDialog(null, "Done");
	}
	public static void processFileText(String text, FileWriter fw, File pdf, String saveDir) {
		// find "methods" at the end of the line in spite of casing, but only if the next line does not start with a lower case letter.
		// you don't want this to find a section like "(see Methods)"
		// and you do not want it to be "methods" in a sentence that is continued on 
		//  the next line: so you look for lower casing after the new line.
		// Numbers are okay with punctuation (example: "3.1 Methods"), but not letters
		//  (section numbers are used sometimes)
		//Pattern methodsWord = Pattern.compile("[\r\n]+(?![a-zA-Z][\\.,\\?\\!\\;]).*([mM]ethod(s)?|METHOD(S)?)[\r\n]+(?![a-z])");
		// updated to have forms of "Experimental Procedure"
		Pattern methodsWord = Pattern.compile("[\r\n]+(?![a-zA-Z][\\.,\\?\\!\\;]).*([mM]ethod(s)?|METHOD(S)?|Experimental [Pp]rocedure(s)?|EXPERIMENTAL PROCEDURE(S)?|Experimental [Dd]esign)[\\s\r\n]+(?![a-z])");

		// the matcher can extract instances of a regular expression from the text (named "text"),
		// which was extracted from the pdf
		// in this case, it uses the methods pattern above.
		Matcher m = null;
		// store the methods per pdf here
		String method = "";
		int methodStart = -1;
		if (text == null || !text.contains(" ")) {
			methodStart = -2;
		}else{
			m = methodsWord.matcher(text);

			// finds headers at most like: 
			//   new line #.#.?letter? (word w/caps) word? word?.? new line + no lower case
			Pattern genericHeader = Pattern.compile("[\r\n]+[0-9\\.]*[a-z]?\\.?\\s?[a-zA-Z\\-\\']*[A-Z]+[a-zA-Z\\-\\']*\\s?([a-zA-Z]*\\s?[a-zA-Z]*)\\.?[\r\n]+(?![a-z])");
			// likely too broad
			//Pattern genericHeader = Pattern.compile("[\r\n]+(?![a-zA-Z][\\.,\\?\\!\\;]).*[a-zA-Z\\-\\']*[A-Z]+[a-zA-Z\\-\\']*\\s?([a-zA-Z]*\\s?[a-zA-Z]*)\\.?[\r\n]+(?![a-z])");
			Matcher h = genericHeader.matcher(text);

			// see how generic header puller works
			String[] search = {"results", "discussion","conclusion", "acknowledgment", "acknowledgement", "references"};

			methodStart = -1;
			int headerEnd = 0;
			boolean searching = false;
			// keep finding headers until there are none.
			// only find methods at first, then switch to finding any header
			while ((searching && h.find(headerEnd)) || (!searching && m.find(headerEnd))) {
				// by default, use the method finder
				Matcher inUse = m;
				// if you already found the method, use the header finder
				if(searching)
					inUse = h;
				// leave a new line character for the next search
				headerEnd = inUse.end()-1;

				// get the header name in lower case
				String header = text.substring(inUse.start(), inUse.end()).toLowerCase();
				//if (pdf.getName().equals("11689307.pdf"))
				//System.out.println(header);
				// if you are searching for headers other than methods..
				if (searching) {
					// go through the list of valid headers
					for (String s : search) {
						if (header.contains(s)) {
							// you found a valid header after finding the method.
							searching = false;

							// add the cleaned method contents to a new line
							//  each line represents a method.
							method += cleanup(text.substring(methodStart, h.start()));
							methodStart = -1;
							break;
						}
					}
				} 
				// the method searcher "m" find had to find something to be here 
				else {
					// saw where the method started for future reference.
					methodStart = m.start();
					searching = true;
					//System.out.println(header);
				}
			}
		}


		// if no methods were found (probably an oddball of labeling)
		// print a message for the pdf in question.
		if (method.isEmpty()) {

			String out = "";
			// if method start is not -1, then it found a method header, but not
			//  a valid header after that.
			if (methodStart >= 0)
				out = pdf.getName()+": Could not find the end of the method.";
			else if (methodStart == -1)
				// otherwise, it couldn't find where to begin.
				out = pdf.getName()+": Could not find methods.";
			else if (methodStart == -2)
				// or, it couldn't even find text.
				out = pdf.getName()+": Document had unreadable text.";
			System.out.println(out);
			try {
				fw.write(out+"\n");
			} catch (IOException e) {
				e.printStackTrace();
			}
		} else {
			// if they are found for this pdf,
			//  then write the file containing the Methods for the pdf
			try {
				// make a file in the program directory based on the pdf name 
				//  without the extension.
				File ftmp = new File(saveDir+pdf.getName().substring(0,pdf.getName().length()-4)+".txt");
				// if the file does not exist, make one.
				// (if it does exist, it is overwritten)
				if (!ftmp.exists()) {
					ftmp.createNewFile();
				}
				// write the method to the file
				FileWriter a = new FileWriter(ftmp);
				a.write(method);
				// finish the write with flush() so that everything is definitely finished writing.
				a.flush();
				a.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

	}
}
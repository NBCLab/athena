// PDF Extract Methods
// v0.2
// Jason Hays
// This program uses Apache PDFBox 1.8.10 to extract the Methods section using regular expressions.
//
// It assumes that the Methods and Results sections are sequential.
// The Method and Results sections' names can vary somewhat in casing.
// The output uses the name of the pdf to create a .txt file in the "ExtractMethods Output" folder.
//  Each line has a methods section (if there are multiple experiments).

package dft;

import java.awt.Frame;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Vector;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.swing.JFileChooser;
import javax.swing.JOptionPane;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDDocumentInformation;
import org.apache.pdfbox.util.PDFTextStripper;

public class ExtractMethodsv02 {
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
	public static void getFiles(File dir, Vector<File> files) {
		// the list of all files (including non pdfs)
		File[] fList = dir.listFiles();
		
		// go through the files one by one.
		for (File f : fList) {
		
			if (f.isFile()) {
				// process files that have pdf extensions (add them to the list)
				if (f.getName().endsWith(".pdf"))
					files.add(f);
			
			// process the contents of sub directories too.
			} else if (f.isDirectory()) {
				getFiles(f, files);
			}
		}
	}
	// the main method is where the program starts
	public static void main(String args[]) {
		// make a window for the file chooser
		Frame f = new Frame();
		JFileChooser fc = new JFileChooser();
		
		// start the program wherever the program is located.
		fc.setCurrentDirectory(new File("./"));
		// only allow directories to be chosen
		fc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
		
		// open the dialog (not using tmp)
		int tmp = fc.showOpenDialog(f);
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
		
		// initialize the list of pdf files in the directory
		Vector<File> pdfList = new Vector<File>();
		// get the files.
		getFiles(dirName, pdfList);

		// go through each pdf found earlier
		for (File pdf : pdfList) {
			System.out.println(pdf.getName());
			PDDocument pddDocument = null;
			try {
				// load the contents as a PDF rather than a generic File
				pddDocument = PDDocument.load(pdf);
			} catch (IOException e) {
				e.printStackTrace();
			}

			// if the document has the PMID in the metadata, you could find it here:
			// none of the documents I downloaded from PubMed have it, though.
			//PDDocumentInformation info = pddDocument.getDocumentInformation();
			//System.out.println(info.getMetadataKeys());
			// "PubMed ID"
			
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
			}
			
			// find "methods" at the end of the line in spite of casing, but only if the next line does not start with a lower case letter.
			// you don't want this to find a section like "(see Methods)"
			// and you do not want it to be "methods" in a sentence that is continued on 
			//  the next line: so you look for lower casing after the new line.
			// Numbers are okay with punctuation (example: "3.1 Methods"), but not letters
			//  (section numbers are used sometimes)
			Pattern methodsWord = Pattern.compile("[\r\n]+(?![a-zA-Z][\\.,\\?\\!\\;]).*([mM]ethod(s)?|METHOD(S)?)[\r\n]+(?![a-z])");

			// the matcher can extract instances of a regular expression from the text (named "text"),
			// which was extracted from the pdf
			// in this case, it uses the methods pattern above.
			Matcher m = methodsWord.matcher(text);

			
			// keep the two patterns for method and results separate to speed up the search and
			// to keep it modular.
			//  The criteria is easier to change when separate patterns are involved.
			
			// find the results section because that's where the methods section ends.
			// can be:
			// Results or "Results and Discussion" of various casing without a letter with punctuation beforehand.
			//Pattern resultsWord = Pattern.compile("[\r\n]+(?![a-zA-Z][\\.,\\?\\!\\;]).*(((Results|RESULTS)(\\s(AND|[Aa]nd)\\s(DISCUSSION|[Dd]iscussion))?))[\r\n]+(?![a-z])");
			Pattern resultsWord = Pattern.compile("[\r\n]+.*(((Results|RESULTS)(\\s(AND|[Aa]nd)\\s(DISCUSSION|[Dd]iscussion))?))[\r\n]+(?![a-z])");
			Matcher r = resultsWord.matcher(text);
			
			// store the methods here
			String method = "";
			// advance the search's start index as you find methods and results headers.
			int start = 0;
			// find each of the Methods (based on start)
			// and only continue if you find one.
			while (m.find(start)) {
				// set the search start at the end of the match.
				start = m.end();
				// find the next, corresponding results section
				if (r.find(start)) {
					// set the search start
					start = r.end();
					// clean the text and add it onto the end of what has been found so far.
					// Each Methods section has its own line.
					method += cleanup(text.substring(m.start(), r.start()))+"\n";
				}
			}
			// if no methods were found (probably an oddball of labeling)
			// print a message for the pdf in question.
			if (method.isEmpty()) {
				try {
					fw.write(pdf.getName()+"\n");
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				System.out.println(pdf.getName()+": Could not find methods");
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
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			// close the pdf
			try {
				pddDocument.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		try {
			fw.flush();
			fw.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		JOptionPane.showMessageDialog(null, "Done");
	}
}
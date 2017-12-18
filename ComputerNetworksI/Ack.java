import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.nio.charset.StandardCharsets;

import java.io.IOException;

import java.util.*;

import java.text.SimpleDateFormat;


public class Ack implements RequestCodes, FolderNames {

	private final int LENGTH = 16;

	private final int DURATION = 4*60/1000; 

	// private static Modem modem;

	private String message = "";

	private String encrypted = "";

	private int FCS = 0;

	public void setMessage(String message){
		this.message = message;
	}

	public void getEncrypted(){

		int start 	= message.indexOf('<');
		int end 	= message.indexOf('>');

		this.encrypted = "";
		for (int i = ++start; i<end; i++){
			this.encrypted += this.message.charAt(i);
		}

		// System.out.println("Encrpted message: " + this.encrypted);

		return;
	}

	public void getFCS(){

		String strFCS = "";
		int start = message.indexOf('>') + 2;

		for (int i=start; i<start+3; i++) {
			strFCS += this.message.charAt(i);
		}

		this.FCS = Integer.parseInt(strFCS);

		// System.out.println("FCS = " + this.FCS);

		return;
	}

	public boolean isEqual() {

		char previous = this.encrypted.charAt(0);

		for (int i=1; i<this.LENGTH; i++){
			// System.out.print(previous);
			previous = (char) ( previous ^ this.encrypted.charAt(i) );
		}

		// System.out.printf("FCS: %d and Encrypted Number = %d\n\n", this.FCS, (int) previous);

		return ( this.FCS == (int) previous) ? true: false;
	}

	public void getData(){
		this.getEncrypted();
		this.getFCS();
		return;
	}

	//Return the ack filename for the specific session
	public String createFilename() {

		String folder = ACK_PATH;
		String timeStamp = new SimpleDateFormat("yyyy_MM_dd_HH_mm").format(new Date());
		String name = "ack-";
		String extension = ".txt";
		return folder + name + timeStamp + extension;
	}

	//Create or append to the speciafied text file
	public void saveToFile(String filename){

		try {
		    final Path path = Paths.get(filename);
		    Files.write(
		    		path,
		    		Arrays.asList(this.message),
		    		// this.message.getBytes(),
		    		StandardCharsets.UTF_8,
		        	Files.exists(path) ? StandardOpenOption.APPEND : StandardOpenOption.CREATE);

		    // Files.write(path, "\n".getBytes(), StandardOpenOption.APPEND);
		} catch (IOException e) {
            e.printStackTrace();
        }
	}

}

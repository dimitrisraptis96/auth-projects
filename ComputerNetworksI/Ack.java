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

	public int requests = 1;

	public int errors = 0;

	// private boolean isEqual = false;

/*
	public Ack () {

		// this.message = "";
		// this.encrypted = "";
		// this.FCS = 0;
		// this.modem = modem;
	}*/

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

		System.out.println("Encrpted message: " + this.encrypted);

		return;
	}

	public void getFCS(){

		String strFCS = "";
		int start = message.indexOf('>') + 2;

		for (int i=start; i<start+3; i++) {
			strFCS += this.message.charAt(i);
		}

		this.FCS = Integer.parseInt(strFCS);

		System.out.println("FCS = " + this.FCS);

		return;
	}

	public boolean isEqual() {

		char previous = this.encrypted.charAt(0);

		for (int i=1; i<this.LENGTH; i++){
			// System.out.print(previous);
			previous = (char) ( previous ^ this.encrypted.charAt(i) );
		}

		System.out.printf("\nFCS: %d and Encrypted Number = %d\n", this.FCS, (int) previous);

		return ( this.FCS == (int) previous) ? true: false;
	}

	/*public void request(String code){
		
		sendRequestCode(code);
		setMessage(getStringPacket());
		this.getEncrypted();
		this.getFCS();
		return;
	}*/

	/*public void resetData(){
		this.encrypted 	= "";
		this.FCS 		= "";
		return;
	}*/

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
	// public void checkRequest(){

	// 	String filename = createFilename();

	// 	long startTime = System.currentTimeMillis();

	// 	this.request(ACK_REQUEST_CODE);
	// 	saveToFile(filename, this.message);
	// 	do {
	// 		if (this.isEqual()){
	// 			request(ACK_REQUEST_CODE);
	// 		}
	// 		else {
	// 			errors++;
	// 			request(NACK_REQUEST_CODE);
	// 		}
	// 		requests++;
	// 		saveToFile(filename, this.message);

	// 	} while( (System.currentTimeMillis() - startTime) > DURATION );

	// 	System.out.printf("Errors = $d\nTotal Requests = %d\n", errors, requests);

	// 	return;

	// }

}

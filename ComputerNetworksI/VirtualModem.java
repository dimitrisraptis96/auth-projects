/*
*
* Computer Networks I
*
* Experimental Virtual Lab
*
* Java virtual modem communications seed code
*
*/ 

import java.io.File;
import java.io.OutputStream;
import java.io.InputStream;
import java.io.FileOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

// import java.nio.file.Files;
// import java.nio.file.Paths;
// import java.nio.file.StandardOpenOption;
import java.nio.charset.StandardCharsets;

import java.util.*;

import java.text.SimpleDateFormat;


interface RequestCodes {

	static final String ECHO_REQUEST_CODE        	= "E1264";
	
	static final String IMAGE_REQUEST_CODE       	= "M1378";
	
	static final String IMAGE_ERROR_REQUEST_CODE 	= "G8166";
	
	static final String GPS_REQUEST_CODE         	= "P6664R=5051299";
	
	static final String ACK_REQUEST_CODE        	= "Q3599";

	static final String NACK_REQUEST_CODE        	= "R5911";
}

interface FolderNames {

	static final String IMAGE_PATH       		= "./data/images/";
	
	static final String IMAGE_ERROR_PATH		= "./data/error_images/";

	static final String GPS_PATH				= "./data/gps/";

	static final String ACK_PATH				= "./data/ack/";

	static final String NACK_PATH				= "./data/nack/";

}


public class VirtualModem implements RequestCodes, FolderNames {

	private static Modem modem;

	private final long SESSION_DURATION = 4*60; 
	
	public static void main(String[] param) {
 		(new VirtualModem()).demo();
 	}

 	//Initialize modem characteristics
 	public void setModem(){

 		modem=new Modem();
		modem.setSpeed(80000);
		modem.setTimeout(6000);
		return;
 	}
 
 	//Send the request codes with return delimiter
 	private boolean sendRequestCode (String requestCode){
 		return modem.write( (requestCode + "\r").getBytes() );
 	}

 	//Read characters from modem and save them to a String that returns
 	private String getStringPacket() {

 		int k;
 		String response = "";

 		while(true) {
			try {

				k=modem.read();
				if (k== -1) break;
				response += (char)k;
			} catch (Exception x) {

				x.printStackTrace();
			}
		}

		System.out.println(response);
		return response;
 	}

 	//Return a unique filename at the specified folder
	public String getFilename (String folder, String extension) {

		String timeStamp = new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss").format(new Date());
		return folder + timeStamp + extension;
	}

    //Get an image from the modem and save it as .JPG file
    public boolean saveFile (String folder, String extension){

        try {
        	int k;

    		OutputStream out = new FileOutputStream (getFilename(folder, extension));

            while( (k = modem.read()) != -1 ) {

					out.write ( k );
			}

            out.close();
	        return true;

        } catch (IOException e) {
            e.printStackTrace();
        	return false;

        }
    }

    //Write the request code and create the appropiate file 
    boolean getPacket(String requestCode, String folder, String extension){

    	if (!sendRequestCode(requestCode)){
 
			System.out.println("Writing to virtualModem failed..\n");
			return false;
		}

		if (saveFile(folder, extension)) {

			System.out.println("File " + extension + " saved sucessfully!!!\n");
			return true;
		}

    	return false;
    }

    //INEFFICIENT way
    //Another way to read modem. Get the entire stream and then read from the local stream. 
    public boolean saveInputStream(){

    	try {

    		int k;

	    	InputStream in = modem.getInputStream();
	   		OutputStream out = new FileOutputStream (getFilename(IMAGE_PATH, ".txt"));
	   		
	   		while ((k = in.read()) != -1) {
	   			out.write(k);
	   		}
	    	
	    	out.close();
	    	in.close();
	    	return true;
    	}
    	catch (IOException e) {
    		e.printStackTrace();
    		return false;
    	}

    }

    public void ackRequest(){

    	Ack ack = new Ack();

		String filename = ack.createFilename();

		long startTime = System.currentTimeMillis();

		sendRequestCode(ACK_REQUEST_CODE);
		ack.setMessage(getStringPacket());
		ack.getData();

		ack.saveToFile(filename);
		do {
			if (ack.isEqual()){
				sendRequestCode(ACK_REQUEST_CODE);
			}
			else {
				sendRequestCode(NACK_REQUEST_CODE);
				ack.errors++;
			}

			ack.setMessage(getStringPacket());
			// ack.resetData();
			ack.getData();
			ack.requests++;
			ack.saveToFile(filename);

		} while( (System.currentTimeMillis() - startTime) / 1000 > SESSION_DURATION );

		//last is equal?
		
		System.out.printf("Errors = %d\nTotal Requests = %d\n", ack.errors, ack.requests);

		return;

	}

	public void demo() {

		this.setModem();
		modem.open("ithaki");
		this.getStringPacket();


		this.ackRequest();
		//Ack testing
		// sendRequestCode(ACK_REQUEST_CODE);
		// new Ack(this.modem).checkRequest();
		// ack.getEncrypted();
		// ack.getFCS();

		// while (ack.isEqual()) {
		// 	System.out.println("No equal!");
		// }


		// long startTime = System.currentTimeMillis();  

		//===============================================================
		//Get different packets
		//===============================================================

		// getPacket(IMAGE_REQUEST_CODE      , IMAGE_PATH 		, ".JPG");
		// getPacket(IMAGE_ERROR_REQUEST_CODE, IMAGE_ERROR_PATH, ".JPG");  
		// getPacket(GPS_REQUEST_CODE		  , GPS_PATH		, ".txt");  
		// getPacket(ACK_REQUEST_CODE		  , ACK_PATH		, ".txt");  
		// getPacket(NACK_REQUEST_CODE		  , NACK_PATH		, ".txt");

		//===============================================================

		// long estimatedTime = System.currentTimeMillis() - startTime;
 
		// System.out.println("Time elapsed: " + Long.toString(estimatedTime) + " msec");

		modem.close();

		System.out.println("\n\nConnection closed.");
	}
}
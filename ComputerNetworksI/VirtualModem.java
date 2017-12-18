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

	static final String ECHO_REQUEST_CODE        	= "E0830";
	
	static final String IMAGE_REQUEST_CODE       	= "M5512";
	
	static final String IMAGE_ERROR_REQUEST_CODE 	= "G0036";
	
	static final String GPS_REQUEST_CODE         	= "P2420R=5051299";
	
	static final String ACK_REQUEST_CODE        	= "Q3021";

	static final String NACK_REQUEST_CODE        	= "R1346";
}

interface FolderNames {

	static final String SESSION_1_PATH			= "./session1/";

	static final String SESSION_2_PATH			= "./session2/";

	static final String ECHO_PATH       		= "./data/echo/";

	static final String IMAGE_PATH       		= "./data/images/";
	
	static final String IMAGE_ERROR_PATH		= "./data/error_images/";

	static final String GPS_PATH				= "./data/gps/";

	static final String ACK_PATH				= "./data/ack/";

	static final String NACK_PATH				= "./data/nack/";

}


public class VirtualModem implements RequestCodes, FolderNames {

	private final long SESSION_DURATION = 1*60*1000;

	private static Modem modem;

	private long responseTime;
	
	public static void main(String[] param) {
 		(new VirtualModem()).demo();
 	}

 	//Initialize modem characteristics
 	public void setModem(){

 		modem=new Modem();
		modem.setSpeed(80000);
		modem.setTimeout(1000);
		return;
 	}
 
 	//Send the request codes with return delimiter
 	private boolean sendRequestCode (String requestCode){
 		return modem.write( (requestCode + "\r").getBytes() );
 	}

 	//Read characters from modem and save them to a String that returns
 	private String getStringPacket() {

 		this.responseTime = System.currentTimeMillis();
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

		this.responseTime = System.currentTimeMillis() - responseTime;
		System.out.println(response);
		return response;
 	}

 	//Return a unique filename at the specified folder
	public String getFilename (String folder, String extension) {

		String timeStamp = new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss").format(new Date());
		return folder + timeStamp + extension;
	}

	//Return a unique filename at the specified folder
	public String getTextFilename (String folder, String name) {

		String timeStamp = new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss").format(new Date());
		String extension = ".text";
		return folder + name + timeStamp + extension;
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



    public void echoRequest(String path){
    	try {
    		int packets = 0;

    		modem.setTimeout(100);

			FileOutputStream out = new FileOutputStream( getTextFilename(path, "echo-") );
			out.write(("Response time of echo packets\n").getBytes());
			out.write(("=============================\n").getBytes());

			long startTime = System.currentTimeMillis();

			do {
				sendRequestCode(ECHO_REQUEST_CODE);
				getStringPacket();
				out.write((this.responseTime + "\n").getBytes());
				packets++;

			} while( SESSION_DURATION > (System.currentTimeMillis() - startTime) );

			out.write(("\n\nTotal Packets = " + packets +"\n").getBytes());
			out.close();

		} catch (IOException e) {
    		e.printStackTrace();
    	}
    }



    public void arqRequest(String path){

		try {
	    	modem.setTimeout(200);

	    	int errors 		= 0;
	    	int requests 	= 1;
	    	int repeats		= 0;
	    	
	    	Arq arq = new Arq();

			FileOutputStream out = new FileOutputStream( getTextFilename(path, "arq-") );
			out.write(("Response time of ARQ transmitted packets!\n").getBytes());
			out.write(("=========================================\n").getBytes());

			long startTime = System.currentTimeMillis();

			sendRequestCode(ACK_REQUEST_CODE);
			arq.setMessage(getStringPacket());
			arq.getData();

			do {
				if (arq.isEqual()){
					out.write((arq.getTime() + " " + this.responseTime + " " + repeats + "\n").getBytes());
					sendRequestCode(ACK_REQUEST_CODE);
					repeats = 0;
				}
				else {
					sendRequestCode(NACK_REQUEST_CODE);
					errors++;
					repeats++;
				}

				arq.setMessage(getStringPacket());
				arq.getData();
				requests++;

			} while( SESSION_DURATION > (System.currentTimeMillis() - startTime) );

			out.write(("\n\nErrors = " + errors + " Total Requests = " + requests +"\n").getBytes());
			out.close();

			System.out.printf("Errors = %d\nTotal Requests = %d\n", errors, requests);

		} catch (IOException e) {
    		e.printStackTrace();
    	}

	}

	public void demo() {

		this.setModem();
		modem.open("ithaki");
		this.getStringPacket();

		//===========================
		//Session requests
		//===========================
		// getPacket(IMAGE_REQUEST_CODE      , SESSION_1_PATH, ".JPG");
		// getPacket(IMAGE_ERROR_REQUEST_CODE, SESSION_1_PATH, ".JPG");
		// this.echoRequest(SESSION_1_PATH);
		// this.arqRequest(SESSION_1_PATH);
		sendRequestCode(GPS_REQUEST_CODE);
		getStringPacket();


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
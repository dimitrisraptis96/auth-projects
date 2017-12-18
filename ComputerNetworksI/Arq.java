import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.nio.charset.StandardCharsets;

import java.io.IOException;

import java.util.*;

import java.text.SimpleDateFormat;


public class Arq implements RequestCodes, FolderNames {

	private final int LENGTH = 16;

	private String message;

	private String encrypted;

	private int FCS;


	public Arq(){

		this.message 	= "";
		this.encrypted 	= "";
		this.FCS 		= 0;
	}

	public void setMessage(String message){
		this.message = message;
	}

	public void getEncrypted(){

		int start 	= message.indexOf('<') + 1;
		int end 	= message.indexOf('>');

		this.encrypted = "";
		for (int i = start; i<end; i++){
			this.encrypted += this.message.charAt(i);
		}
	}

	public void getFCS(){

		String strFCS = "";

		int start 	= message.indexOf('>') + 2;
		int end 	= start + 3;

		for (int i=start; i<end; i++) {
			strFCS += this.message.charAt(i);
		}
		//Convert string to integer
		this.FCS = Integer.parseInt(strFCS);
	}

	public String getTime(){

		String time = "";

		int start 	= message.indexOf(':') - 2;
		int end 	= start + 8;

		for (int i=start; i<end; i++) {
			time += this.message.charAt(i);
		}
		return time;
	}

	public boolean isEqual() {

		char previous = this.encrypted.charAt(0);

		for (int i=1; i<this.LENGTH; i++){
			previous = (char) ( previous ^ this.encrypted.charAt(i) );
		}

		return ( this.FCS == (int) previous) ? true: false;
	}

	public void getData(){

		this.getEncrypted();
		this.getFCS();
	}
}

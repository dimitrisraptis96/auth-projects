import java.util.*;


public class Arq {

	private final int LENGTH = 16;

	private String message;

	private String encrypted;

	private int FCS;


	public Arq(){

		this.message 	= "";
		this.encrypted 	= "";
		this.FCS 		= 0;
	}

	//Set the ACK or NACK message
	public void setMessage(String message){
		this.message = message;
	}

	//Get the encrypted code of the packet
	public void setEncrypted(){

		int start 	= message.indexOf('<') + 1;
		int end 	= message.indexOf('>');

		this.encrypted = "";
		for (int i = start; i<end; i++){
			this.encrypted += this.message.charAt(i);
		}
	}

	//Get the FCS code of the packet
	public void setFCS(){

		String strFCS = "";

		int start 	= message.indexOf('>') + 2;
		int end 	= start + 3;

		for (int i=start; i<end; i++) {
			strFCS += this.message.charAt(i);
		}
		//Convert string to integer
		this.FCS = Integer.parseInt(strFCS);
	}

	//Get the time of the packet's transmission
	public String getTime(){

		String time = "";

		int start 	= message.indexOf(':') - 2;
		int end 	= start + 8;

		for (int i=start; i<end; i++) {
			time += this.message.charAt(i);
		}
		return time;
	}

	//Check if FCS and encrypted number are equal
	public boolean isEqual() {

		char previous = this.encrypted.charAt(0);

		for (int i=1; i<this.LENGTH; i++){
			previous = (char) ( previous ^ this.encrypted.charAt(i) );
		}

		return ( this.FCS == (int) previous) ? true: false;
	}

	//Setting encrypted and FCS variables
	public void setData(){

		this.setEncrypted();
		this.setFCS();
	}
}

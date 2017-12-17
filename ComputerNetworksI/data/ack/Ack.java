import java.util.*;


public class Ack {

	private final int LENGTH = 16;

	private final String message;

	private String encrypted;

	private int FCS;

	// private boolean isEqual = false;


	public Ack () {

		this.message = "";
		this.encrypted = "";
		this.FCS = 0;
	}

	public Ack (String message) {

		this.message = message;
		this.encrypted = "";
		this.FCS = 0;
	}

	public void getEncrypted(){

		int start 	= message.indexOf("<")++;
		int end 	= message.indexOf(">");

		for (int i = start; i<end; i++){
			this.enrypted += this.message.charAt(i);
		}

		System.out.println(this.encrypted);

		return;
	}

	public void getFCS(){

		String strFCS = "";
		int start = message.indexOf(">") + 2;

		for (int i=start; i<start+3; i++) {
			strFCS += this.message.charAt(i);
		}

		this.FCS = Integer.parseInt(strFCS);

		System.out.println(this.FCS);

		return;
	}

	public boolean isEqual() {

		char previous = this.encrypted.charAt(0);

		for (int i=1; i<this.LENGTH; i++){
			System.out.print(previous);
			previous = (char) ( previous ^ encrypted.charAt(i) );
		}

		System.out.printf("\nFCS: %d and Encrypted Number = %d\n", this.FCS, (int) previous);

		return ( this.FCS == (int) previous) ) ? true: false;
	}

	public static void main (String[] args) {

		getEncrypted();
		getFCS();

		while (!isEqual()) {
			System.out.println("No equal!");
		}
		
		return;
	}
}

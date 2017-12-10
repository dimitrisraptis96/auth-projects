import java.util.*;


public class ack {

	public static void main (String[] args) {

		String encrypted = "ddPdnvWarowasgtJ";
		int fcs = 004;

		char previous = encrypted.charAt(0);

		for (int i=1; i<15; i++){
			System.out.print(previous);
			previous = (char) ( previous ^ encrypted.charAt(i) );
		}

		System.out.printf("\nFinal number: %d", (int) previous);

		return;
	}
}
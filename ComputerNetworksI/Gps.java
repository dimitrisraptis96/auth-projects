import java.util.*;

public class Gps {

	private final int DURATION = 4;

	private String message;

	private String time;

	private String latitude;	//mhkos

	private String longitude;	//platos

	public Gps(){
		message = time = latitude = longitude = "";
	}

	public void setMessage(String message){
		this.message = message;
	}

	public String[] getLines(String str){
		return str.split(System.getProperty("line.separator"));
	}

	public String buildParamT (){

		this.longitude = this.longitude.substring(0,4) + this.longitude.substring(5,7);
		this.latitude  = this.latitude.substring(1,5) + this.latitude.substring(6,8);
		return "T=" + this.latitude + this.longitude;
	}

	public ArrayList<String> getParamTList(){

		int previousTime = 0;

		String[] content;
		String[] lines = getLines(this.message);

		ArrayList<String> paramTList = new ArrayList<String>();

		for (String line: lines){
			System.out.println(line);
			System.out.println(line.substring(0,5));
			if( line.substring(0,5).equals("START") ) continue;
			if( line.substring(0,4).equals("STOP")  ) continue;
			content = line.split(",");
			System.out.printf("content length = %d", content.length);

			if (content.length == 0){
				System.out.println("Gps content is empty!");
				return paramTList;
			}

			this.time 		= content[1];
			this.longitude 	= content[2];
			this.latitude 	= content[4];

			//check if 4 sec ellapsed
			if (Integer.parseInt(this.time.substring(0,6)) - previousTime < 4) continue;

			paramTList.add(buildParamT());
			//Set new base time
			previousTime = Integer.parseInt(this.time.substring(0,6));
		}
		return paramTList;
	}	
}

// START ITHAKI GPS TRACKING	
// $GPGGA,102523.000,4038.2013,N,02256.0490,E,1,09,1.1,14.1,M,36.1,M,,0000*63
// $GPGGA,102524.000,4038.1984,N,02256.0515,E,1,09,1.1,14.4,M,36.1,M,,0000*69
// $GPGGA,102525.000,4038.1954,N,02256.0541,E,1,09,1.1,14.4,M,36.1,M,,0000*64
// $GPGGA,102526.000,4038.1923,N,02256.0568,E,1,09,1.1,14.5,M,36.1,M,,0000*6D
// $GPGGA,102527.000,4038.1890,N,02256.0596,E,1,09,1.1,14.5,M,36.1,M,,0000*64
// $GPGGA,102528.000,4038.1857,N,02256.0624,E,1,09,1.1,14.4,M,36.1,M,,0000*6B
// $GPGGA,102529.000,4038.1823,N,02256.0650,E,1,09,1.1,14.3,M,36.1,M,,0000*6D
// $GPGGA,102530.000,4038.1789,N,02256.0676,E,1,09,1.1,14.3,M,36.1,M,,0000*6E
// $GPGGA,102531.000,4038.1753,N,02256.0702,E,1,09,1.1,14.2,M,36.1,M,,0000*6B
// $GPGGA,102532.000,4038.1714,N,02256.0729,E,1,09,1.1,14.0,M,36.1,M,,0000*60
// STOP ITHAKI GPS TRACKING
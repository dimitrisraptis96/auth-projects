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

	//Get the entire gps message
	public void setMessage(String message){
		this.message = message;
	}

	//Return an String[] array including the lines of the message
	public String[] getLines(String str){
		return str.split(System.getProperty("line.separator"));
	}

	//Convert latitude and longitude from DM format to DMS
	public String convertDM2DMS (String DM){

		int tmp = Integer.parseInt(DM) * 60;
		return Integer.toString(tmp).substring(0,2);
	}

	//Build the T parameter
	public String buildParamT (){

		this.longitude = this.longitude.substring(0,4) + convertDM2DMS(this.longitude.substring(5,7));
		this.latitude  = this.latitude.substring(1,5)  + convertDM2DMS(this.latitude.substring(6,8));
		return "T=" + this.latitude + this.longitude;
	}

	//Get the T param within every line of the message
	public ArrayList<String> getParamTList(){

		int previousTime = 0;

		String[] content;
		String[] lines = getLines(this.message);

		ArrayList<String> paramTList = new ArrayList<String>();

		for (String line: lines){
			if( line.substring(0,5).equals("START") ) continue;
			if( line.substring(0,4).equals("STOP")  ) continue;
			content = line.split(",");

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
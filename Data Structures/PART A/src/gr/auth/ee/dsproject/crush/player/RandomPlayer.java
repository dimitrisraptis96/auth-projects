package gr.auth.ee.dsproject.crush.player;

import gr.auth.ee.dsproject.crush.CrushUtilities;
import gr.auth.ee.dsproject.crush.board.Board;

import java.util.ArrayList;

/**
 * <p>
 * Class: Random Player
 * </p>
 * 
 * <p>
 * Description: This class creates the player of the game,.
 * </p>
 * 
 * @author Raptis Dimitrios & Papageorgiou Thomas
 */

public class RandomPlayer implements AbstractPlayer
{
  private int id;		//indicates if the player is blur or red
  private String name;	//indicates the name of the player
  private int score;	//indicates the number of the candies the player has collected
  	
  // Constructors
  
  //Default Constructor
  public RandomPlayer (){
	  
  }
  
  /**
   * Constructor
   * 
   * @param  pid 	this {@code Integer} variable indicates the unique code for the player. 
   */
  public RandomPlayer (Integer pid){
	  this.id = pid.intValue();		//pid is Integer, so we have to get the int value with intValue() method in order to initialize the id
  }

  
  // Setters
  public void setId (int id){
	  this.id = id;		//this.id represents the class id variable and id the Constructor's parameter
  }
  
  public void setName (String name){
	  this.name = name;
  }
  
  public void setScore (int score){
	  this.score = score;
  }
  
  
  // Getters
  public int getId (){
	  return id;
  }
  
  public String getName (){
	  return name;
  }
  
  public int getScore (){
	  return score;
  }

  /**
   * int[] getNextMove (ArrayList<int[]> availableMoves, Board board)
   * 
   * @param  availableMoves 	{@code int[]} array that contains all the available moves the player has.
   * @param  board  			{@code Board} variable that represents the playable board of the game.
   * @return moves				return an int array ( int[4] ) that represents the next random move of the player
   */
  public int[] getNextMove (ArrayList<int[]> availableMoves, Board board)
  {
	
	  int randomIndex =(int) ( Math.random()*( availableMoves.size() ) ); 	//choose a random move from the available ones
	  																		//availableMoves.size() return the size of the array
	  																		//Math.random()*( availableMoves.size() get one random
	  
	  int [] nextMove = new int[3]; 
	  
	  nextMove = CrushUtilities.getRandomMove (availableMoves, randomIndex); //get next move 
	  
	  int [] moves = new int[4];	//moves[0] --> x1
	  								//moves[1] --> y1
	  								//moves[2] --> x2
	  								//moves[3] --> y2
	  
	  moves[0] = nextMove[0];
	  moves[1] = nextMove[1];
	  int dir = nextMove[2];
	  
	  if (dir == CrushUtilities.UP){
		  moves[2] = nextMove[0];
		  moves[3] = nextMove[1] + 1;	//if the direction is UP we have to increase y coord by 1
	  }
	  
	  if (dir == CrushUtilities.DOWN){
		  moves[2] = nextMove[0];
		  moves[3] = nextMove[1] - 1;	//if the direction is DOWN we have to decrease y coord by 1
	  }
	  
	  if (dir == CrushUtilities.RIGHT){
		  moves[2] = nextMove[0] + 1;
		  moves[3] = nextMove[1];		//if the direction is RIGHT we have to increase x coord by 1
	  }
	  
	  if (dir == CrushUtilities.LEFT){
		  moves[2] = nextMove[0] - 1;	//if the direction is LEFT we have to increase x coord by 1
		  moves[3] = nextMove[1];
	  }
	  
	  return moves;
  }

}

package gr.auth.ee.dsproject.crush.board;

/**
 * <p>
 * Class: Tile
 * </p>
 * 
 * <p>
 * Description: This class creates the tiles of the board, that have the candies.
 * </p>
 * 
 * @author Raptis Dimitrios & Papageorgiou Thomas
 */

public class Tile
{

  protected int id;
  private int x;
  private int y;
  private int color;
  private boolean mark;

  /**
   * Default constructor
   */
  public Tile (){
	  
  }
  
  /**
   * Constructor
   * 
   * @param  id 	this {@code int} variable indicates the unique code for each tile.
   * @param  x  	this {@code int} variable indicates the location of the tile on the x-axis.
   * @param  y 		this {@code int} variable indicates the location of the tile on the y-axis.
   * @param  color	this {@code int} variable indicates the candy's color of the tile. 
   * @param  mark 	this {@code boolean} represents whether the tile is selected or not to clear 
   * 				the game board in the next move because participation in triple, quartet or 
   * 				quintet of candies with same color.
   */
  public Tile (int id, int x, int y, int color, boolean mark){
	  this.id = id; //this.id represents the class id variable and id the Constructor's parameter
	  this.x = x;	
	  this.y = y;
	  this.color = color;
	  this.mark = mark;
  }
  
  
  //SETTERS
  
  /**
   * Setter for id: Initialize the class variable id.
   * 
   * @param  id 	this {@code int} variable indicates the unique code for each tile.
   */
  public void setId (int id){
	  this.id = id;
  }
  
  /**
   * Setter for x: Initialize the class variable x.
   * 
   * @param  x  	this {@code int} variable indicates the location of the tile on the x-axis.
   */
  public void setX (int x){
	  this.x = x;
  }
  
  /**
   *Setter for y: Initialize the class variable y.
   * 
   * @param  y  	this {@code int} variable indicates the location of the tile on the y-axis.
   */
  public void setY (int y){
	  this.y = y;
  }
  
  /**
   *Setter for color: Initialize the class variable color.
   * 
   * @param  color  	this {@code int} variable indicates the candy's color of the tile. 
   */
  public void setColor(int color){
	  this.color = color;
  }
  
  /**
   *Setter for mark: Initialize the class variable mark.
   * 
   * @param  mark  	this {@code boolean} represents whether the tile is selected or not to clear 
   * 				the game board in the next move because participation in triple, quartet or 
   * 				quintet of candies with same color. 
   */
  public void setMark(boolean mark){
	  this.mark = mark;
  }
  
  
  //GETERS
  
  /**
   *Getter for id
   * 
   * @return  id 	the {@code int} variable id. 
   */
  public int getId (){
	  return id;
  }

  /**
   *Getter for x
   * 
   * @return  x 	the {@code int} variable x. 
   */
  public int getX (){
	  return x;
  }

  /**
   *Getter for y
   * 
   * @return  y 	the {@code int} variable y. 
   */
  public int getY(){
	  return y;
  }
  
  /**
   *Getter for color
   * 
   * @return  color 	the {@code int} variable color. 
   */
  public int getColor(){
	  return color;
  }
  
  /**
   *Getter for mark
   * 
   * @return  mark 	the {@code boolean} variable mark. 
   */
  public boolean getMark(){
	  return mark;
  }
  
}
  

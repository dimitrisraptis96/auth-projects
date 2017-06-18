package gr.auth.ee.dsproject.crush.player;

import java.util.ArrayList;

import gr.auth.ee.dsproject.crush.board.Board;
import gr.auth.ee.dsproject.crush.board.CrushUtilities;
import gr.auth.ee.dsproject.crush.board.Tile;
import gr.auth.ee.dsproject.crush.defplayers.AbstractPlayer;

public class HeuristicPlayer implements AbstractPlayer
{
	  // TODO Fill the class code.

	  int score;
	  int id;
	  String name;

	  public HeuristicPlayer (Integer pid)
	  {
	    id = pid;
	    score = 0;
	  }

	  @Override
	  public String getName ()
	  {

	    return "evaluation";

	  }

	  @Override
	  public int getId ()
	  {
	    return id;
	  }

	  @Override
	  public void setScore (int score)
	  {
	    this.score = score;
	  }

	  @Override
	  public int getScore ()
	  {
	    return score;
	  }

	  @Override
	  public void setId (int id)
	  {

	    this.id = id;

	  }

	  @Override
	  public void setName (String name)
	  {

	    this.name = name;

	  }

	  @Override
	  public int[] getNextMove (ArrayList<int[]> availableMoves, Board board)
	  {

	    int[] move = availableMoves.get(findBestMoveIndex(availableMoves, board));

	    return calculateNextMove(move);

	  }

	  int[] calculateNextMove (int[] move)
	  {

	    int[] returnedMove = new int[4];

	    if (move[2] == CrushUtilities.UP) {
	      // System.out.println("UP");
	      returnedMove[0] = move[0];
	      returnedMove[1] = move[1];
	      returnedMove[2] = move[0];
	      returnedMove[3] = move[1] + 1;
	    }
	    if (move[2] == CrushUtilities.DOWN) {
	      // System.out.println("DOWN");
	      returnedMove[0] = move[0];
	      returnedMove[1] = move[1];
	      returnedMove[2] = move[0];
	      returnedMove[3] = move[1] - 1;
	    }
	    if (move[2] == CrushUtilities.LEFT) {
	      // System.out.println("LEFT");
	      returnedMove[0] = move[0];
	      returnedMove[1] = move[1];
	      returnedMove[2] = move[0] - 1;
	      returnedMove[3] = move[1];
	    }
	    if (move[2] == CrushUtilities.RIGHT) {
	      // System.out.println("RIGHT");
	      returnedMove[0] = move[0];
	      returnedMove[1] = move[1];
	      returnedMove[2] = move[0] + 1;
	      returnedMove[3] = move[1];
	    }
	    return returnedMove;
	  }
	  
	  int findBestMoveIndex (ArrayList<int[]> availableMoves, Board board)
	  {	
		  //System.out.println("========BOARD BEFORE==========");
		  //board.showBoard();
		  
		  double[] evals = new double[availableMoves.size()]; 
		  
		  // Call  moveEvaluation() for "availableMoves.size()" times to evaluate every available move user has
		  for (int i=0; i < availableMoves.size(); i++){
			  
			  evals[i] = moveEvaluation( availableMoves.get(i), board );
		  }
		  
		  int max = 0;	// Set randomly initial value for compare
		  // Find max value of evals[] table
		  for (int i=1; i<availableMoves.size(); i++){
			  if (evals[i] > evals[max])
				  max = i;
		  }
		  
		  //System.out.println("Best move index: "+max);
		  return max;
	  }

	  double moveEvaluation (int[] move, Board board)
	  {
		double evaluation = 0;
		Board newBoard = CrushUtilities.boardAfterFirstMove(board, move);
		
		//If deleted candies are vertical, it's more possible to make n-ples
		evaluation += deletedCandiesOrientation (move, newBoard);
		//System.out.println("Vertical(0.5) or Horizontal(1) evaluation: " + evaluation);

		//Same color candies in a table width=3 height=3 
		evaluation += calculateSameColorInProximityCandies (move, newBoard) / 10;
		//System.out.println("Same color candies nearby evaluation(number/10): " + evaluation);
		
		//Get height of deleted candies (candy that it's closer to y = 0)
		evaluation += calculateHeight (move,newBoard);
		//System.out.println("Height (closer candy to y = 0) evaluation: " + evaluation);
		
		//Get deleted candies ( * 2 because it's the most important criterion
		evaluation += deletedCandies(move,newBoard) * 2;
		//System.out.println("Deleted candies evaluation : " + evaluation);
		
		//Get deleted candies after chain moves (*2 because it should have the same weight as deletedCandies() )
		evaluation += chainMoves (newBoard) * 2;
		//System.out.println("Deleted chain candies evaluation (final): " + evaluation);
		
		return evaluation;
	  }
	  
	  //========================================================
	  //All System.out.println comments were used for debugging 
	  //	and weren't deleted because they may be useful
	  //========================================================
	  
	  //Give width and height around initialTile(move[0],move[1])
	  //		ex. | | | |4| | | |  width = 3
	  //Return an int [7] array with the number of tile's color in proximity 
	  //		ex. color[2] = 5 means I have 5 blue candies
	  int [] sameColorInProximity (int width, int height, int [] move, Board board){
		  
		  int widthLeft, widthRight;	//help avoiding get OutOfBound error
		  widthLeft = widthRight = width;
		  int heightUp, heightDown;	
		  heightUp = heightDown = height;
		  
		  //Check if width or height is not appropriate and adjust them
		  if (move[0] < width){
			  widthLeft = move[0];
		  }
		  if (CrushUtilities.NUMBER_OF_COLUMNS - move[0] - 1 < width){ // -1 because 'CrushUtilities.NUMBER_OF_COLUMNS' = 10 and I need 9
			  widthRight = CrushUtilities.NUMBER_OF_COLUMNS - move[0] - 1;
		  }
		  if (move[1] < height){
			  heightDown = move[1];
		  }
		  if (CrushUtilities.NUMBER_OF_PLAYABLE_ROWS - move[1] - 1 < height){
			  heightUp = CrushUtilities.NUMBER_OF_PLAYABLE_ROWS - move[1] - 1;
		  }
		  
		  //Fill the coloNumber[] array with the number of every color 
		  //7 int table because we have 7 colors
		  int colorNumber [] = new int [7]; 
		  
		  for (int x = move[0] - widthLeft; x < move[0] + widthRight + 1; x++){
			  
			  for (int y = move[1] - heightDown; y < move[1] + heightUp + 1; y++){
				  colorNumber [board.giveTileAt (x, y).getColor()] ++;  
			  }
		  }
		  
		  //System.out.println ("widthRight = " + widthRight + " " + " heightUp = " + heightUp +" " + " widthLeft = " + widthLeft +" " + " heightDown = " + heightDown);
		  //System.out.println("Color in proximity:");
		  //for (int i=0; i<7; i++){
		  //System.out.println(colorNumber[i]+" ");}
		  
		  return colorNumber;
	  }
	  
	  //Calculate the number of candies that are same as the 2 switching candies, 
	  //because only these two colors may make a n-ple  
	  //Return a double value of these points
	  double calculateSameColorInProximityCandies (int [] move, Board board){
		  
		  int[] colors = new int[7];
		  int sameColorCandies;
		  
		  colors = sameColorInProximity(3, 3, move, board);
		  
		  sameColorCandies = colors [board.giveTileAt (move[0], move[1]).getColor()];	//Calculate candies same as initial tile of move
		  
		  //Calculate candies same as destination tile of move
		  if (move[2] == CrushUtilities.UP) {
			  sameColorCandies += colors [board.giveTileAt (move[0], move[1] + 1).getColor()];
		  }
		  if (move[2] == CrushUtilities.DOWN) {
			  sameColorCandies += colors [board.giveTileAt (move[0], move[1] - 1).getColor()];
		  }
		  if (move[2] == CrushUtilities.RIGHT) {
			  sameColorCandies += colors [board.giveTileAt (move[0] + 1, move[1]).getColor()];
		  }
		  if (move[2] == CrushUtilities.LEFT) {
			  sameColorCandies += colors [board.giveTileAt (move[0] - 1, move[1]).getColor()];
		  }
		  
		  //System.out.println ("Same color candies(same as the 2 switching candies):" + sameColorCandies);
		  return  sameColorCandies;
	  }

	  //Get the y number of a single row 
	  //Return an ArrayList<Tile> with the deleted Tiles
	  ArrayList<Tile> deletedCandiesAtRow (Board board, int row){
		  //Can be found up to two n-ples (max 6 candies)
		  
		  ArrayList<Tile> sameTiles = new ArrayList<Tile>();
		  ArrayList<Tile> deletedTiles = new ArrayList<Tile>();
		  
		  int n_pleOffset = 0;//if n candies are deleted, we have to skip n 'for' loops
		  
		  for (int x = 0; x < CrushUtilities.NUMBER_OF_COLUMNS - 2; x++){ 
			  // For's end value is "CrushUtilities.NUMBER_OF_COLUMNS - 2" because we don't need to check 2 last tiles
			  
			  //Skips loops if i had same candies before.
			  //	Ex. if i had 4 same candies |2|2|2|2|4|5| 
			  //		the next 'for' loop will start after the same candies (candie's color '4')
			  if (n_pleOffset > 0){ 
				  n_pleOffset--;
				  continue;
			  }
			  int compareIndex = 1;	//compareIndex: index that add to x value in order to get the next tile
			  sameTiles.add (board.giveTileAt (x, row)); 
			  
			  //Compare the initial Tile (x,row) with the following tiles, if comparison is true go for the next one 
			  //in order to find the same candies 
			  //Also check if (x + compareIndex) is valid (into the board)
			  while ( (x + compareIndex) < CrushUtilities.NUMBER_OF_COLUMNS  &&  sameTiles.get(0).getColor() == board.giveTileAt(x + compareIndex, row).getColor() && sameTiles.get(0).getColor() != -1 ){
				  sameTiles.add (board.giveTileAt (x + compareIndex, row));
				  compareIndex++;	  
			  }
			  //Check if I have a matching n-ple (n >= 3)
			  if (sameTiles.size() > 2){
				  for (int i=0; i < sameTiles.size(); i++){
					  deletedTiles.add (sameTiles.get(i));
				  }
				  n_pleOffset = sameTiles.size();	//Set skipping loops
			  }
			  sameTiles.clear();
			  compareIndex = 1;
		  }
		  
		  //System.out.println("Deleted Candies at row " + row + " :" + deletedTiles.size());
		  return deletedTiles;
	  }
	  
	  //Get the x number of a single column 
	  //Return an ArrayList<Tile> with the deleted Tiles
	  ArrayList<Tile> deletedCandiesAtColumn (Board board, int column){
		  //Can be found up to two n-ples (max 6 candies)

		  ArrayList<Tile> sameTiles = new ArrayList<Tile>();
		  ArrayList<Tile> deletedTiles = new ArrayList<Tile>();
		  
		  int n_pleOffset = 0;//if n candies are deleted, we have to skip n 'for' loops
		  
		  for (int y = 0; y < CrushUtilities.NUMBER_OF_PLAYABLE_ROWS - 2; y++){ 
			  // For's end value is "CrushUtilities.NUMBER_OF_PLAYABLE_ROWS - 2" because we don't need to check 2 last tiles
			  
			  //Skips loops if i had same candies before.
			  //Ex. if i had 4 same candies |2|
			  //							|2|
			  //							|2|
			  //							|2|
			  //							|4|
			  //							|5|
			  //	the next 'for' loop will start after the same candies (candie's color '4')
			  if (n_pleOffset > 0){ 
				  n_pleOffset--;
				  continue;
			  }
			  int compareIndex = 1;	//compareIndex: index that add to x value in order to get the next tile
			  sameTiles.add (board.giveTileAt (column, y)); 
			  
			  //Compare the initial Tile (column, y) with the following tiles, if comparison is true go for the next one 
			  //in order to find the same candies 
			  //Also check if (x + compareIndex) is valid (into the board)
			  while ( (y + compareIndex) < CrushUtilities.NUMBER_OF_COLUMNS  &&  sameTiles.get(0).getColor() == board.giveTileAt(column, y + compareIndex).getColor() && sameTiles.get(0).getColor() != -1 ){
				  sameTiles.add (board.giveTileAt (column, y + compareIndex));
				  compareIndex++;	  
			  }
			  //Check if I have a matching n-ple (n >= 3)
			  if (sameTiles.size() > 2){
				  for (int i=0; i < sameTiles.size(); i++){
					  deletedTiles.add (sameTiles.get(i));
				  }
				  n_pleOffset = sameTiles.size();
			  }
			  sameTiles.clear();
			  compareIndex = 1;
		  }
		  
		  //System.out.println("Deleted Candies at column " + column + " :" + deletedTiles.size());
		  return deletedTiles;
	  }

	  //Calculate the deleted candies of the board
	  //Return the number of the deleted candies 
	  int deletedCandies (int[] move, Board board){
		  //DEBUG
		  //System.out.println("========BOARD AFTER==========");
		  //board.showBoard();
		  //System.out.println("Tile x:" + move[0]);
		  //System.out.println("Tile y:" + move[1]); //test comment delete
		  
		  //System.out.println("Tile color:"+board.giveTileAt(move[0],move[1]).getColor());
		  //System.out.println("move: " + move[0] + " " + move[1] + " " + move[2]);
		  
		  int candies = 0;

		  //We don't have to check all the board for same candies
		  //Only possible triple will be at 2 rows and 1 column (UP-DOWN) or at 2 column and 1 row (RIGHT-LEFT)
		  if (move[2] == CrushUtilities.UP) {
		      //System.out.println("UP");
			  candies = deletedCandiesAtColumn( board, move [0] ).size();
			  candies += deletedCandiesAtRow ( board, move [1] ).size();
			  candies += deletedCandiesAtRow ( board, move [1] + 1 ).size();
		  }
		  if (move[2] == CrushUtilities.DOWN) {
		      //System.out.println("DOWN");
			  candies = deletedCandiesAtColumn ( board, move [0] ).size();
			  candies += deletedCandiesAtRow ( board, move [1] ).size();
			  candies += deletedCandiesAtRow ( board, move [1] - 1 ).size();
		  }
		  if (move[2] == CrushUtilities.RIGHT) {
		      //System.out.println("RIGHT");
			  candies = deletedCandiesAtRow ( board, move [1] ).size();
			  candies += deletedCandiesAtColumn ( board, move [0] ).size();
			  candies += deletedCandiesAtColumn ( board, move [0] + 1 ).size();
		  }
		  if (move[2] == CrushUtilities.LEFT) {
		      //System.out.println("LEFT");
			  candies = deletedCandiesAtRow ( board, move [1] ).size();
			  candies += deletedCandiesAtColumn ( board, move [0] ).size();
			  candies += deletedCandiesAtColumn ( board, move [0] - 1 ).size();
		  }
		  
		  //System.out.println("Deleted candies after move:" + candies);
		  return candies;
	  }
	  
	  //Return true if the deleted candies are horizontal
	  boolean areDeletedCandiesHorizontal (int[] move, Board board){
		  
		  int candies = 0;
		  //Check only the possible rows that will have a n-ple
		  if (move[2] == CrushUtilities.UP) {
			  candies += deletedCandiesAtRow ( board, move [1] ).size();
			  candies += deletedCandiesAtRow ( board, move [1] + 1 ).size();
		  }
		  if (move[2] == CrushUtilities.DOWN) {
			  candies += deletedCandiesAtRow ( board, move [1] ).size();
			  candies += deletedCandiesAtRow ( board, move [1] - 1 ).size();
		  }
		  if (move[2] == CrushUtilities.RIGHT || move[2] == CrushUtilities.LEFT) {
			  candies = deletedCandiesAtRow ( board, move [1] ).size();
		  }
		  
		  //System.out.println("Horizontal Candies:" + candies);
		  return candies > 0;
	  }
	  
	  //Return true if the deleted candies are vertical
	  boolean areDeletedCandiesVertical (int[] move, Board board){
		  
		  int candies = 0;
		  //Check only the possible columns that will have a n-ple
		  if (move[2] == CrushUtilities.UP || move[2] == CrushUtilities.DOWN) {
			  candies = deletedCandiesAtColumn ( board, move [0] ).size();
		  }
		  if (move[2] == CrushUtilities.RIGHT) {
			  candies += deletedCandiesAtColumn ( board, move [0] ).size();
			  candies += deletedCandiesAtColumn ( board, move [0] + 1 ).size();
		  }
		  if (move[2] == CrushUtilities.LEFT) {
			  candies += deletedCandiesAtColumn ( board, move [0] ).size();
			  candies += deletedCandiesAtColumn ( board, move [0] - 1 ).size();
		  }
		  
		  //System.out.println("Vertical Candies:" + candies);
		  return candies > 0;
	  }
	  
	  //Find out the orientation of the deleted candies
	  //Return : 	1    if there are horizontal 
	  //			0.5	 if there are vertical
	  //			0	 never, because we have always deleted candies 
	  //				 	(if avalaibleMoves is empty, moveEvaluation() won't be called)
	  // * If move is horizontal chain moves are more possible
	  double deletedCandiesOrientation (int[] move, Board board){
		  
		  if ( areDeletedCandiesHorizontal (move, board) ){
			  return 1;
		  }
		  else if ( areDeletedCandiesVertical (move, board) ){
			  return 0.5; 
		  }
		  else{
			  return 0;
		  }
		  
	  }

	  //Calculate the height of the deleted candies
	  //* Height: the row of the candy that it's closer to y = 0 row 
	  //ex.			|0|
	  //			|0|
	  //	y = 2	|0|		
	  //			|3|
	  //	y = 0	|2|		So height = 2
	  int heightOfDeletedCandies (int move[], Board board){
		  
		  int y1 = CrushUtilities.NUMBER_OF_PLAYABLE_ROWS; //give a max value in order to get the right value if the returned array is empty
		  int y2 = CrushUtilities.NUMBER_OF_PLAYABLE_ROWS; 

		  ArrayList<Tile> sameTilesArray = new ArrayList<Tile>();
		  
		  if ( areDeletedCandiesHorizontal(move,board) ){
		  //Deleted candies are Horizontal (See: areDeletedCandiesHorizontal() method)
		  			if (move[2] == CrushUtilities.UP) {
		  				//Have to find out which row have the deleted candies and return the min of them
		  				if ( deletedCandiesAtRow ( board, move [1] ).size() > 0 ) 		return move [1];
		  				if ( deletedCandiesAtRow ( board, move [1] + 1 ).size() > 0 ) 	return move [1] + 1;
		  			}
		  			if (move[2] == CrushUtilities.DOWN) {
		  				//Have to find out which row have the deleted candies and return the min of them
		  				if ( deletedCandiesAtRow ( board, move [1] - 1 ).size() > 0 ) 	return move [1] - 1;
		  				if ( deletedCandiesAtRow ( board, move [1] ).size() > 0 ) 		return move [1];
		  			}
		  			if (move[2] == CrushUtilities.RIGHT || move[2] == CrushUtilities.LEFT) {
		  				//Deleted candies are at the move's row
		  				return move[1];
		  			}
		  }
		  
		  //Vertical move is far more difficult..
		  //The reason deletedCandiesAtColumn or Row returning an array
		  if ( areDeletedCandiesVertical (move, board) ){
		  //Deleted candies are Vertical (See: areDeletedCandiesVertical() method)
		  			if (move[2] == CrushUtilities.RIGHT) {
		  				//Have to check 2 column and select the min candie of them
		  				sameTilesArray = deletedCandiesAtColumn( board, move [0] );
		  				if ( sameTilesArray.size() > 0){
		  					//if sameTilesArray is empty then y1 = 10 and y2 < 10, so return value is y2
		  					y1 = sameTilesArray.get(0).getY();
		  				}
		  				sameTilesArray = deletedCandiesAtColumn( board, move [0] + 1 );
		  				if ( sameTilesArray.size() > 0){
		  					//if sameTilesArray is empty then y2 = 10 and y1 < 10, so return value is y1
		  					y2 = sameTilesArray.get(0).getY();
		  				}
		  				
		  				//Want to return the min value of y1-y2
		  				if (y1 < y2) return y1;
		  				return y2;
		  			}
		  			if (move[2] == CrushUtilities.LEFT) {
		  				sameTilesArray = deletedCandiesAtColumn( board, move [0] );
		  				if ( sameTilesArray.size() > 0){
		  					//if sameTilesArray is empty then y1 = 10 and y2 < 10, so return value is y2
		  					y1 = sameTilesArray.get(0).getY();
		  				}
		  				sameTilesArray = deletedCandiesAtColumn( board, move [0] - 1 );
		  				if ( sameTilesArray.size() > 0){
		  					//if sameTilesArray is empty then y2 = 10 and y1 < 10, so return value is y1
		  					y2 = sameTilesArray.get(0).getY();
		  				}
		  				
		  				//want to return the min value of y1-y2
		  				if (y1 < y2) return y1;
		  				return y2;
		  			}
		  			
		  			if (move[2] == CrushUtilities.UP || move[2] == CrushUtilities.DOWN) {
		  				//Just return the lowest candy's y value
		  				return deletedCandiesAtColumn( board, move [0] ).get(0).getY();
		  			}
		  }
		  //Will never go here! (because we have always deleted candies. If avalaibleMoves is empty, moveEvaluation() won't be called)			
		  return -1;	
	  }
		
	  //Calculate and return the points (double) will be awarded to move because of it's height (See: heightOfDeletedCandies() )
	  double calculateHeight(int[] move, Board board){
			double height = heightOfDeletedCandies(move,board);
			//System.out.println("Height: " + height);
			if 		(height > 8)		height = - 1.5;
			else if (height > 6)		height = -1;
			else if (height > 3)		height = 0;
			else if (height > 0)		height = 1;
			else if (height == 0)		height = 2;
			else height = 0;
		  	return height;
	  }
	  
	  int chainMoves (Board board){
		  	int candies = 0;
		  
		  	Board newBoard = CrushUtilities.boardAfterDeletingNples(board);
			  
		  	//System.out.println("========BOARD AFTER FIRST CRUSH==========");
		  	//newBoard.showBoard();
		  	
	  		//Check the board for horizontal candies, to find the candies deleted from chain moves 
		  	for (int y = 0; y < CrushUtilities.NUMBER_OF_PLAYABLE_ROWS; y++){
		  		candies += deletedCandiesAtRow (newBoard, y).size() ;
		  	}
		  	
	  		//Check the board for vertical candies, to find the candies deleted from chain moves 
		  	for (int x = 0; x < CrushUtilities.NUMBER_OF_COLUMNS; x++){
		  		candies += deletedCandiesAtColumn (newBoard, x).size() ;
		  	}
		  	
		  	//System.out.println("Chain moves Candies:" + candies);
		  	return candies;  	
	  }
	  
	  
}
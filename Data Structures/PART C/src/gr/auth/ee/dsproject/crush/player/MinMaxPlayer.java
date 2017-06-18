package gr.auth.ee.dsproject.crush.player;

import gr.auth.ee.dsproject.crush.board.Board;
import gr.auth.ee.dsproject.crush.board.CrushUtilities;
import gr.auth.ee.dsproject.crush.defplayers.AbstractPlayer;
import gr.auth.ee.dsproject.crush.node.Node84678577;

import java.util.ArrayList;

public class MinMaxPlayer implements AbstractPlayer
{
  // TODO Fill the class code.

  int score;
  int id;
  String name;

  public MinMaxPlayer (Integer pid)
  {
    id = pid;
    score = 0;
  }

  public String getName ()
  {

    return "MinMax";

  }

  public int getId ()
  {
    return id;
  }

  public void setScore (int score)
  {
    this.score = score;
  }

  public int getScore ()
  {
    return score;
  }

  public void setId (int id)
  {
    this.id = id;
  }

  public void setName (String name)
  {
    this.name = name;
  }

  //========================================================
  //All System.out.println comments were used for debugging 
  //	and weren't deleted because they may be useful
  //========================================================
  
  
  //Choose the best move 
  //Return the move as output of calculateNextMove() function
  public int[] getNextMove (ArrayList<int[]> availableMoves, Board board)
  {
	Board copiedBoard = CrushUtilities.cloneBoard(board);
	
	//Initiate the root of my tree
	Node84678577 root = new Node84678577 (0, copiedBoard);
	
	//Create the whole tree
	createMySubTree (root, 1);
	
	//DEBUG
	//showTree(root);
	
	//Find the index of the best move
    int indexBest = chooseMove (root); 
    
    //DEBUG
    //showTree(root);
    
    //Get the best move
    int[] bestMove = availableMoves.get(indexBest);

    return CrushUtilities.calculateNextMove(bestMove);
    
  }

  //Create the subtree for the min-max player. 
  private void createMySubTree (Node84678577 parent, int depth)
  {
	  //Get Node's available moves 
	  ArrayList<int[]> availableMoves = new ArrayList<int[]> ();    
	  availableMoves = CrushUtilities.getAvailableMoves( parent.getNodeBoard() );
	  
	  //Create a sub-node for every available move 
	  for (int i=0; i < availableMoves.size(); i++){ 
		  
		  //DEBUG
		  //System.out.println("%%%%% MOVE NUMBER:"+i+"%%%%%");//test
		  
		  //Get board after applying the move (n-ples are not deleted!!!)
		  Board boardAfterMove = new Board();
		  boardAfterMove = CrushUtilities.boardAfterFirstMove(parent.getNodeBoard(), availableMoves.get(i));
		  //Initiate tempNode
		  Node84678577 tempNode = new Node84678577 (parent, depth, availableMoves.get(i), boardAfterMove);

		  //Calculate node's evaluation
		  if (this.getScore() > 300) {
			  //Check if my player has won
			  //System.out.println("MY PLAYER WINS");
			  
			  tempNode.setNodeEvaluation ( Double.POSITIVE_INFINITY );
		  }
		  else{
			//Create an Heuristic Player in order to use it's evaluation method
			HeuristicPlayer heurPlayer = new HeuristicPlayer(this.getId());
			tempNode.setNodeEvaluation ( heurPlayer.moveEvaluation(availableMoves.get(i), parent.getNodeBoard()) );
		  }
		  
		  //System.out.println("Node's #"+i+" evaluation (depth = "+depth+") : "+tempNode.getNodeEvaluation());
		  
		  parent.addChild (tempNode);
		  
		  //Create tempNode's subtree
		  createOpponentSubTree (tempNode, depth + 1);	
	  }
  }

  //Create the subtree for my opponent
  private void createOpponentSubTree (Node84678577 parent, int depth)
  {

	  Node84678577 grandParent = parent.getParent();//If depth = 2 grandParent node is the root of the tree
	  
	  //Get the initial state of the board, in order to apply the node's move.
	  Board initialBoard = new Board();
	  initialBoard = CrushUtilities.cloneBoard ( grandParent.getNodeBoard() );
	  
	  //DEBUGGING
	  //System.out.println("==========INITIAL BOARD=============");
	  //initialBoard.showBoard();//test
	  
	  //Get the fullBoard after applying the move, in order to get the opponent's available moves
	  Board fullBoard = new Board();
	  fullBoard = CrushUtilities.boardAfterFullMove( initialBoard, parent.getNodeMove() );
	  
	  //DEBUGGING
	  //System.out.println("==========FULL MOVE BOARD=============");
	  //fullBoard.showBoard();
	  
	  ArrayList<int[]> availableMoves = new ArrayList<int[]> ();
	  availableMoves = CrushUtilities.getAvailableMoves( fullBoard );
	  
	  //Create a sub-node for every available move 
	  for (int i=0; i < availableMoves.size(); i++){ 

		  Board boardAfterMove = CrushUtilities.boardAfterFirstMove(fullBoard, availableMoves.get(i));
		  Node84678577 tempNode = new Node84678577 (parent, depth, availableMoves.get(i), boardAfterMove);

		  //Negative evaluation of the node
		  if (this.getScore() > 300) {
			  //Check if opponent has won
			  //System.out.println("OPPONENT PLAYER WINS");
			  
			  tempNode.setNodeEvaluation ( Double.NEGATIVE_INFINITY );
		  }
		  else{
			  //Create an Heuristic Player in order to use it's evaluation method
			  HeuristicPlayer heurPlayer = new HeuristicPlayer(this.getId());
			  tempNode.setNodeEvaluation ( - heurPlayer.moveEvaluation(availableMoves.get(i), parent.getNodeBoard()) );
		  }
		  
		  //System.out.println("Node's #"+i+" evaluation (depth = "+depth+") : "+tempNode.getNodeEvaluation());

		  parent.addChild(tempNode);
	  }
  }

  //Use my AB-Pruning algorithm in order to find the index of the best available move 
  //Get the root of the tree as argument
  //Return the index of the best available move
  private int chooseMove (Node84678577 root)
  {
	  ArrayList<Node84678577> children = new ArrayList<Node84678577> ();
	  children = root.getChildren();
	  
	  for (int i=0; i < children.size(); i++){
		  //Add evaluation of the root's children to the root's grand-children
		  addEvaluationToChildren ( children.get(i) );
		  
		  //Unnecessary!!! Set root's evaluation to zero(0).
		  children.get(i).setNodeEvaluation(0);
		  
		  //Minimum evaluation of grand-children sets child's evaluation.
		  children.get(i).setNodeEvaluation( findMinEvaluationOfChildren ( children.get(i) ) );	  
	  }
	  
	  //Find max evaluation of children and set root's evaluation
	  root.setNodeEvaluation (findMaxEvaluationOfChildren (root));
	  
	  //DEBUGGING
	  //System.out.println("Index of Best move:"+findIndexOfEvaluation(root, root.getNodeEvaluation()));
	  
	  //Find the index of the best evaluation
	  return findIndexOfEvaluation(root, root.getNodeEvaluation());
  }
	
  //Add evaluation of the node given to its children's evaluation
  //Get a node 
  private void addEvaluationToChildren (Node84678577 root){
	  
	  ArrayList<Node84678577> children = new ArrayList<Node84678577> ();
	  children = root.getChildren();
	  
	  for (int i = 0; i < children.size(); i++){
		  //Add root's evaluation to its children.
		  children.get(i).setNodeEvaluation( children.get(i).getNodeEvaluation() + root.getNodeEvaluation() );
	  }
  }
  
  //Check node's children in order to find the minimum evaluation of them.
  //Get the node
  //Return the minimum evaluation value
  private double findMinEvaluationOfChildren (Node84678577 root){
	  
	  ArrayList<Node84678577> children = new ArrayList<Node84678577> ();
	  children = root.getChildren();
	  
	  double min = Double.POSITIVE_INFINITY;
	  
	  //Find minimum evaluation of the root's children.
	  for (int i=0; i < children.size(); i++){
		  if ( children.get(i).getNodeEvaluation() < min) {
			  min = children.get(i).getNodeEvaluation();
		  }
	  }
	  return min;
  }
 
 //Check node's children in order to find the maximum evaluation of them.
 //Get the node  
 //Return the maximum evaluation value
 private double findMaxEvaluationOfChildren (Node84678577 root){
	  
	  ArrayList<Node84678577> children = new ArrayList<Node84678577> ();
	  children = root.getChildren();
	  
	  double max = Double.NEGATIVE_INFINITY;
	  
	  //Find maximum evaluation of the root's children.
	  for (int i=0; i < children.size(); i++){
		  if ( children.get(i).getNodeEvaluation() > max) {
			  max = children.get(i).getNodeEvaluation();
		  }
	  }
	  return max;
  }
 
 //Search node's children in order to find the evaluation that is equal to a given double value
 //		and return its index.
 //Get the node and a evaluation value.
 //Return the index of the child with the requested evaluation
 //		or -1 if the evaluation is not found.
 private int findIndexOfEvaluation (Node84678577 root, double evaluation){
	  
	  ArrayList<Node84678577> children = new ArrayList<Node84678577> ();
	  children = root.getChildren();
	  
	  //Find the index of the children's evaluation that is equal to "evaluation" variable
	  for (int i=0; i < children.size(); i++){
		  
		  if ( children.get(i).getNodeEvaluation() == evaluation) {		  
			  System.out.println("Index of Best move:"+i);//test
			  return i;
		  }
	  }
	 
	  //If no evaluation matchs return -1
	  return -1;
 }
 
 //FOR DEBUGGING
 //Prints the whole tree's evaluation values.
 //Works right ONLY if we give root as argument and tree's depth is 2.
 public void showTree (Node84678577 root){
	 
	 System.out.println("==========THE TREE=============");
	 System.out.println(root.getNodeEvaluation());
	 
	 ArrayList<Node84678577> children = new ArrayList<Node84678577> ();
	 children = root.getChildren();
	 
	 for (int i=0; i < children.size(); i++){
		 System.out.print(children.get(i).getNodeEvaluation()+"     ");
	 }
	 System.out.println();
	 
	 for (int i=0; i < children.size(); i++){
		for (int j=0; j < children.get(i).getChildren().size(); j++){
			 System.out.print(children.get(i).getChildren().get(j).getNodeEvaluation()+"/");
		}
		System.out.print("      ");
		}
		System.out.println();
 }	
 
}
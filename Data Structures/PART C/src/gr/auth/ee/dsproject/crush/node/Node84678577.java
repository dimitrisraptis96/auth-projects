package gr.auth.ee.dsproject.crush.node;

import gr.auth.ee.dsproject.crush.board.Board;

import java.util.ArrayList;


public class Node84678577
{
  // TODO Rename and fill the code
	
	Node84678577 parent;
	ArrayList<Node84678577> children;
	int nodeDepth;
	int [] nodeMove;
	Board nodeBoard;
	double nodeEvaluation;
	
	
	
	//Constructor for the root
	public Node84678577 (int nodeDepth, Board nodeBoard){
		this.children = new ArrayList<Node84678577>();
		this.nodeDepth = nodeDepth;
		this.nodeBoard = nodeBoard;
	}
	
	//Constructor for the other nodes
	public Node84678577 (Node84678577 parent, int nodeDepth, int [] nodeMove, Board nodeBoard){
		this.children = new ArrayList<Node84678577>();
		this.parent = parent;
		this.nodeDepth = nodeDepth;
		this.nodeMove = nodeMove;
		this.nodeBoard = nodeBoard;
	}
	
	
	
	//Setters
	public void setParent (Node84678577 parent){
		this.parent = parent;
	}
	
	public void setNodeMove (int[] move){
		this.nodeMove = move;
	}
	
	public void setNodeDepth (int depth){
		this.nodeDepth = depth;
	}
	
	public void setNodeBoard (Board board){
		this.nodeBoard = board;
	}
	
	public void setNodeEvaluation (double evaluation){
		this.nodeEvaluation = evaluation;
	}
	
	
	
	//Getters
	public Node84678577 getParent (){
		return parent;
	}
	
	public int[] getNodeMove (){
		return nodeMove;
	}
	
	public int getNodeDepth (){
		return nodeDepth;
	}
	
	public Board getNodeBoard (){
		return nodeBoard;
	}
	
	public double getNodeEvaluation (){
		return nodeEvaluation;
	}
	
	public ArrayList<Node84678577> getChildren (){
		return children;
	}
	
	
	
	//Methods
	
	//Unnecessary
	public boolean isRoot(){
		return (nodeDepth == 0);
	}
	
	//Add a child to the node
	public void addChild (Node84678577 child){
		this.children.add(child);
	}
}

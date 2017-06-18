package gr.auth.ee.dsproject.crush.player;

import gr.auth.ee.dsproject.crush.board.Board;

import java.util.ArrayList;

public interface AbstractPlayer
{
  public void setId (int id);

  public int getId ();

  public void setName (String name);

  public String getName ();

  public void setScore (int score);

  public int getScore ();

  public int[] getNextMove (ArrayList<int[]> availableMoves, Board board);

}

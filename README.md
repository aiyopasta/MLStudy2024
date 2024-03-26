"computation_graph.py" contains, well, the computation graph structure. I should've really made a parent class "Node" 😂

Noteable implementations:
(1) 2D Decision Boundary Visualization: See MLP_playground.py & vert_db.glsl + frag_db.glsl. If you change the network architecture in the .py file, do it in the fragment shader too!
(2) "Grand Tour" MNIST training visualization. Note: It might completely glitch out because of the number of pygame surfaces on screen. If so, rerun it. There's likely a more efficient way to do this. It's also interactive, in principle. Drag the red number handles around!

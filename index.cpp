#include<iostream>
#include<list>
#include<vector>
#include <stack>
using namespace std;
class graph{
    int vertices;
    list<int> *adj;
    public:
    graph(int v){
        vertices = v;
        adj = new list<int>[vertices];
    }
    void addEdges(int u,int v){
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    void Dfs(int start,vector<int> &dfs){
         vector<int>vis(vertices,0);
         vis[start]=1;
         stack<int> s;
         s.push(start);
         while(!s.empty()){
            int node = s.top();
            s.pop();
            dfs.push_back(node);
            for(auto it:adj[node]){
                if(!vis[it]){
                    vis[it]=1;
                    s.push(it);
                }
            }
         }
    }
};

// dfs
int main(){
    int vertices,edges;
    cout << "Enter the no of vertices in the graph";
    cin>>vertices;
    cout << "Enter the no of edges in the graph";
    cin>>edges;
    graph g(vertices);
    cout<<"Enter the edges (u,v):" << endl;
    for(int i =0;i<edges;i++){
        int u,v;
        cin >> u >> v;
        g.addEdges(u,v);
    }
    int start;
    cout << "Enter the starting node of the traversal";
    cin >> start;
    vector<int> dfs;
    g.Dfs(start,dfs);
    for(auto i : dfs){
        cout << i;
    }
    return 0;

}
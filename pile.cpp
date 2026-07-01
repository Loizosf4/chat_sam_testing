#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    cin>>n;
    int a[n];
    for(int i=0;i<n;i++){
        cin>>a[i];
    }
    int total=0;
    for(int i=n-2;i>=0;i--){
        int tmp=i;
        vector <int> v;
        for(int j=i+1;j<n;j++){
            if(a[i]>a[j])
                v.push_back(j);
        }
        for(int j=v.size()-1;j>=0;j--){
            if(a[i]>a[v[j]]){
                int diafora=a[i]-a[v[j]];
                a[v[j]]+=diafora;
                a[i]-=diafora;
                total+=diafora;
            }
            else
                break;
        }

    }
    return 0;
}
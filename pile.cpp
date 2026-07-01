#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    cin>>n;
    int a[n];
    int b[n];
    for(int i=0;i<n;i++){
        cin>>a[i];
        b[i]=a[i];
    }
    int total=0,total2=0;
    for(int i=n-1;i>=1;i--){
        if(b[i]<=b[i-1]){
            b[i]--;
            break;
        }
    }

    for(int i=n-2;i>=0;i--){
        int tmp=i;
        vector <int> v;
        vector <int> v2;
        for(int j=i+1;j<n;j++){
            if(a[i]>a[j])
                v.push_back(j);
            if(b[i]>b[j])
                v2.push_back(j);        
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
        for(int j=v.size()-1;j>=0;j--){
            if(b[i]>b[v2[j]]){
                int diafora=b[i]-b[v2[j]];
                b[v2[j]]+=diafora;
                b[i]-=diafora;
                total2+=diafora;
            }
            else
                break;
        }
    }
    
    cout<<max(total,total2);
    return 0;
}
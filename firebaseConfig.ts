// firebaseConfig.ts
// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyACmYnt_UCaJ9yUR6TE7YmNXdAAQWNP5Fk",
  authDomain: "thirdspace-1a419.firebaseapp.com",
  databaseURL: "https://thirdspace-1a419-default-rtdb.firebaseio.com",
  projectId: "thirdspace-1a419",
  storageBucket: "thirdspace-1a419.appspot.com",
  messagingSenderId: "488388834009",
  appId: "1:488388834009:web:0469b537247e83015a44c8",
  measurementId: "G-47ZGPVEWZN"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
//const database = getDatabase(app);

//export { database };
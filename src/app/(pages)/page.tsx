import Link from "next/link";
import Toolbar from "../modules/Toolbar";
//
import { Metadata } from "next";

export const metadata: Metadata = {
  title: 'Home',
  description: '...',
}

/** Business Users: Name, Email, Password?, Catagory */
/** Users: Name, Email, Password?, Preferred Catagory, Price, Location */
export default function Home() {
  return (
    <div className="bg-[#f4f4f4]">
      <Toolbar />
      <div className="py-20 flex flex-col items-center justify-center">
        <img src="/Phantom.svg" alt="Third Space Logo" style={{height: "200px", width: "400px", 
          objectFit: "cover", objectPosition: "center"}} />
        <h2 className="text-center py-3 text-3xl font-bold">Welcome to Third Space!</h2>
        <h2 className="text-center">Your one stop shop for all your local eating needs!</h2>
        <br></br>
        <Link href="/foryou"><button className="bg-black text-white py-2 px-4 rounded hover:bg-gray-700">
          <p className="text-white">Let&apos;s Get Started!</p>
        </button></Link>
        <br></br>
        <Link href="/mybusiness"><button className="bg-black text-white py-2 px-4 rounded hover:bg-gray-700">
          <p className="text-white">I&apos;m A Business And I Want To Join!</p>
        </button></Link>
      </div>
    </div>
  );
}
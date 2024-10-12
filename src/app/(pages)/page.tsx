import Link from "next/link";
import Toolbar from "../modules/Toolbar";
import { Button, Input } from "@mui/material";
//
import { Metadata } from "next";

export const metadata: Metadata = {
  title: 'Home',
  description: '...',
}

export default function Home() {
  return (
    <div className="the-fancy-background">
      <Toolbar/>
        <div className="py-11 flex flex-col items-center justify-center">
            <h2 className="text-center py-3 text-3xl">Food Tinder</h2>
            <ChatInput />
        </div>

        <br/>
        <h1 className="font-bold text-xl">Explore Buisnesses:</h1>
    </div>
  );
}

const ChatInput = () => {

  return (
      <div>
          <Input 
              placeholder="Talk about yourself!" />&nbsp;
          <Link href="/foryou"><Button className="bg-black text-white">Let's Get Started!</Button></Link>
      </div>
  );
};

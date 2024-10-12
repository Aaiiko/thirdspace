import Link from "next/link";
import { BuisnessCard } from "../modules/BuisnessCard";
import Toolbar from "../modules/Toolbar";
import { Button, Input } from "@mui/material";
//
import { Metadata } from "next";

export const metadata: Metadata = {
  title: 'Home',
}

export default function Home() {
  return (
    <div>
      <Toolbar/>
        <div className="py-11 flex flex-col items-center justify-center">
            <h2 className="text-center py-3">Hey wassup welcome to thirdspace (Insert Logo Here)</h2>
            
        </div>
        <br/>
        <h1 className="font-bold text-xl">Explore Buisnesses:</h1>
        <BuisnessCard name="A Really Cool Buisness" />
    </div>
  );
}


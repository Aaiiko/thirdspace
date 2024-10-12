"use client"

import { useState } from "react";
import { BusinessCard } from "./BusinessCard";
import { Button } from "@mui/material";


interface BusinessExploreProps {
    /** The collection of business for the user to iterate through */
    businesses: Business[];
}

export const BusinessExplore = (props: BusinessExploreProps) => {
    /** The current business the user's looking at */
    const [currentB, setCurrentB] = useState<number>(0);
    /** The businesses they HATE BOOO!!!! BOOOO!!!! */
    const [rejected, setRejected] = useState<Business[]>([]);
    /** The business they LOVE YIPPIE :D GOODY goody GUMDROPS!!! */
    const [loved, setLoved] = useState<Business[]>([]);

    /** Right swipe function (they love!) */
    const rightSwipe = () => {
        setLoved([...loved, props.businesses[currentB]]);
        if (currentB === props.businesses.length - 1) {
            setCurrentB(-1);
        } else {
            setCurrentB(currentB + 1);
        }
    }

    /** Left swipe function (BOOOO!!!!) */
    const leftSwipe = () => {
        setRejected([...rejected, props.businesses[currentB]]);
        if (currentB === props.businesses.length - 1) {
            setCurrentB(-1);
        } else {
            setCurrentB(currentB + 1);
        }
    }

    /** Once user has completed swiping */
    if (currentB === -1 || props.businesses.length <= 0) {
        return (
            <div>
                <h1>Wow! You explored all of the business in the area!</h1>
                <h2 className="font-bold text-4xl">You loved:</h2>
                {loved.map((business) => (
                    <BusinessCard business={business} />
                ))}
                <h2 className="font-bold text-4xl">You rejected:</h2>
                {rejected.map((business) => (
                    <BusinessCard business={business} />
                ))}
            </div>
        )
    }

    /** Shows current business */
    return (
        <div>
            <BusinessCard business={props.businesses[currentB]} />
            <div className="button-container flex justify-between">
                <div>
                    <Button onClick={leftSwipe} className="bg-black text-white">Left Swipe</Button>
                </div>
                <div>
                    <Button onClick={rightSwipe} className="bg-black text-white">Right Swipe</Button>
                </div>
            </div>
        </div>

    );
}
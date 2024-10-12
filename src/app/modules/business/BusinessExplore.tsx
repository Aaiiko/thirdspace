"use client"


import React, { useState, useEffect, useRef } from "react";
import { BusinessCard } from "./BusinessCard";
import { Button } from "@mui/material";


interface BusinessExploreProps {
    /** The collection of business for the user to iterate through */
    businesses: Business[];
}

export const BusinessExplore = (props: BusinessExploreProps) => {
    const [currentB, setCurrentB] = useState<number>(0);
    const [rejected, setRejected] = useState<Business[]>([]);
    const [loved, setLoved] = useState<Business[]>([]);
    const touchStartX = useRef<number>(0);
    const touchEndX = useRef<number>(0);

    /** The event that happens on a right swipe */
    const rightSwipe = () => {
        setLoved([...loved, props.businesses[currentB]]);
        if (currentB === props.businesses.length - 1) {
            setCurrentB(-1);
        } else {
            setCurrentB(currentB + 1);
        }
    };

    /** The event that happens on a left swipe yo */
    const leftSwipe = () => {
        setRejected([...rejected, props.businesses[currentB]]);
        if (currentB === props.businesses.length - 1) {
            setCurrentB(-1);
        } else {
            setCurrentB(currentB + 1);
        }
    };

    const handleTouchStart = (e: React.TouchEvent) => {
        touchStartX.current = e.targetTouches[0].clientX;
    };

    const handleTouchMove = (e: React.TouchEvent) => {
        touchEndX.current = e.targetTouches[0].clientX;
    };

    const handleTouchEnd = () => {
        if (touchStartX.current - touchEndX.current > 50) {
            leftSwipe();
        }

        if (touchEndX.current - touchStartX.current > 50) {
            rightSwipe();
        }
    };

    /** After all businesses have been filtered through or there are none avalible  */
    if (currentB === -1 || props.businesses.length <= 0) {
        return (
            <div className="px-1 py-2">
                <h1>You've seen all the businesses in your area!</h1>
                <p className="font-bold text-4xl">Loved</p>
                {loved.map((business) => (
                    <BusinessCard business={business} />
                ))}
                <p className="font-bold text-4xl">Rejected</p>
                {rejected.map((business) => (
                    <BusinessCard business={business} />
                ))}
            </div>
        );

    }

    return (
        <div
            className="relative min-h-screen"
            onTouchStart={handleTouchStart}
            onTouchMove={handleTouchMove}
            onTouchEnd={handleTouchEnd}
        >
            <div className="p-8 bg-white shadow-lg rounded-lg max-w-md w-full">
                <h2 className="text-2xl font-bold mb-4">{props.businesses[currentB]?.name}</h2>
                <p className="text-gray-700 mb-6">{props.businesses[currentB]?.description}</p>
            </div>
            <div className="button-container flex justify-between sticky bottom-0 left-0 right-0 p-4 bg-white">
                <div>
                    <button onClick={leftSwipe} className="bg-black text-white py-2 px-4 rounded hover:bg-gray-700">
                        Left Swipe
                    </button>
                </div>
                <div>
                    <button onClick={rightSwipe} className="bg-black text-white py-2 px-4 rounded hover:bg-gray-700">
                        Right Swipe
                    </button>
                </div>
            </div>
        </div>
    );
};